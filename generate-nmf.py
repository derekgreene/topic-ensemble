#!/usr/bin/env python
"""
Applies randomly initialized NMF to the specified dataset.

Sample usage:
python generate-nmf.py sample.pkl -k 4 -r 30 --maxiters 100 -o base-nmf
"""
import os, sys, random
import logging as log
from optparse import OptionParser
import numpy as np
import scipy.sparse
import text.util, unsupervised.nmf, unsupervised.rankings, unsupervised.util

# --------------------------------------------------------------

def main():
	parser = OptionParser(usage="usage: %prog [options] dataset_file")
	parser.add_option("--seed", action="store", type="int", dest="seed", help="initial random seed", default=1000)
	parser.add_option("-k", action="store", type="int", dest="k", help="number of topics", default=5)
	parser.add_option("-r","--runs", action="store", type="int", dest="runs", help="number of runs", default=1)
	parser.add_option("--maxiters", action="store", type="int", dest="maxiter", help="maximum number of iterations", default=100)
	parser.add_option("-s", "--sample", action="store", type="float", dest="sample_ratio", help="sampling ratio of documents to include in each run (range is 0 to 1). default is all", default=1.0)
	parser.add_option("--nndsvd", action="store_true", dest="use_nndsvd", help="use nndsvd initialization instead of random")
	parser.add_option("-o","--outdir", action="store", type="string", dest="dir_out", help="base output directory (default is current directory)", default=None)
	parser.add_option('-d','--debug',type="int",help="Level of log output; 0 is less, 5 is all", default=3)
	(options, args) = parser.parse_args()
	if len(args) < 1:
		parser.error( "Must specify at least one corpus file" )	
	log_level = max(50 - (options.debug * 10), 10)
	log.basicConfig(level=log_level, format='%(asctime)-18s %(levelname)-10s %(message)s', datefmt='%d/%m/%Y %H:%M',)

	if options.dir_out is None:
		dir_out_base = os.getcwd()
	else:
		dir_out_base = options.dir_out
		if not os.path.exists(dir_out_base):
			os.makedirs(dir_out_base)		

	# Set random state
	random_seed = options.seed
	if random_seed < 0:
		random_seed = random.randint(1,100000)
	np.random.seed( random_seed )
	random.seed( random_seed )			
	log.info("Using random seed %s" % random_seed )
				
	# Load the cached corpus
	corpus_path = args[0]
	(X,terms,doc_ids,classes) = text.util.load_corpus( corpus_path )
	log.debug( "Read %s document-term matrix, dictionary of %d terms, list of %d document IDs" % ( str(X.shape), len(terms), len(doc_ids) ) )
	
	# Choose implementation
	if options.use_nndsvd:
		init_strategy = "nndsvd"
	else:
		init_strategy = "random"
	impl = unsupervised.nmf.SklNMF( max_iters = options.maxiter, init_strategy = init_strategy )

	n_documents = X.shape[0]
	n_sample = int( options.sample_ratio * n_documents )
	indices = np.arange(n_documents)

	log.info( "Applying NMF (k=%d, runs=%d, seed=%s, init_strategy=%s) ..." % ( options.k, options.runs, options.seed, init_strategy ) )
	if options.sample_ratio < 1:
		log.info( "Sampling ratio = %.2f - %d/%d documents per run" % ( options.sample_ratio, n_sample, n_documents ) )
	log.debug( "Results will be written to %s" % dir_out_base )
	# Run NMF
	for r in range(options.runs):
		log.info( "NMF run %d/%d (k=%d, max_iters=%d)" % (r+1, options.runs, options.k, options.maxiter ) )
		file_suffix = "%s_%03d" % ( options.seed, r+1 )
		# randomly sub-sample the data
		if options.sample_ratio < 1:
			log.info("Subsamping the data ...")
			np.random.shuffle(indices)
			sample_indices = indices[0:n_sample]
			S = X[sample_indices,:]
			log.info("Creating sparse matrix ...")
			S = scipy.sparse.csr_matrix(S)
			sample_doc_ids = []
			for doc_index in sample_indices:
				sample_doc_ids.append( doc_ids[doc_index] )
		else:
			S = X
			sample_doc_ids = doc_ids
		# apply NMF
		log.info("Applying NMF to matrix of size %d X %d ..." % ( S.shape[0], S.shape[1] ) ) 
		impl.apply( S, options.k )
		# Get term rankings for each topic
		term_rankings = []
		for topic_index in range(options.k):		
			ranked_term_indices = impl.rank_terms( topic_index )
			term_ranking = [terms[i] for i in ranked_term_indices]
			term_rankings.append(term_ranking)
		log.debug( "Generated ranking set with %d topics covering up to %d terms" % ( len(term_rankings), unsupervised.rankings.term_rankings_size( term_rankings ) ) )
		# Write term rankings
		ranks_out_path = os.path.join( dir_out_base, "ranks_%s.pkl" % file_suffix )
		log.debug( "Writing term ranking set to %s" % ranks_out_path )
		unsupervised.util.save_term_rankings( ranks_out_path, term_rankings )
		# Write document partition
		partition = impl.generate_partition()
		partition_out_path = os.path.join( dir_out_base, "partition_%s.pkl" % file_suffix )
		log.debug( "Writing document partition to %s" % partition_out_path )
		unsupervised.util.save_partition( partition_out_path, partition, sample_doc_ids )			
		# Write the complete factorization
		factor_out_path = os.path.join( dir_out_base, "factors_%s.pkl" % file_suffix )
		# NB: need to make a copy of the factors
		log.debug( "Writing factorization for %d documents to %s" % ( len(sample_doc_ids), factor_out_path ) )
		unsupervised.util.save_nmf_factors( factor_out_path, np.array( impl.W ), np.array( impl.H ), sample_doc_ids, terms )

	log.info("Generated %d ensemble members" % (r+1))

# --------------------------------------------------------------

if __name__ == "__main__":
	main()
