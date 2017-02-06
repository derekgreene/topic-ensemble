#!/usr/bin/env python
"""
Evaluate the stability (i.e. agreement) between a set of partitions generated on the same dataset, using Pairwise Normalized Mutual Information (PNMI).

Sample usage:
python eval-partition-stability.py models/base/*partition*.pkl 
"""
import os, sys
import logging as log
from optparse import OptionParser
import numpy as np
from prettytable import PrettyTable
from sklearn.metrics.cluster import normalized_mutual_info_score
import unsupervised.nmf, unsupervised.util

# --------------------------------------------------------------

def main():
	parser = OptionParser(usage="usage: %prog [options] partition_file1|directory1 ...")
	parser.add_option("-o","--output", action="store", type="string", dest="out_path", help="path for CSV summary file (by default this is not written)", default=None)
	parser.add_option("--hist", action="store", type="string", dest="hist_out_path", help="path for histogram CSV file (by default this is not written)", default=None)
	# Parse command line arguments
	(options, args) = parser.parse_args()
	if( len(args) < 1 ):
		parser.error( "Must specify one or more partitions/directories" )
	log.basicConfig(level=20, format='%(message)s')

	# Get list of all specified partition files
	file_paths = []
	for path in args:
		if not os.path.exists( path ):
			log.error("No such file or directory: %s" % path )
			sys.exit(1)
		if os.path.isdir(path):
			log.debug("Searching %s for partitions" % path )
			for dir_path, dirs, files in os.walk(path):
				for fname in files:
					if fname.startswith("partition") and fname.endswith(".pkl"):
						file_paths.append( os.path.join( dir_path, fname ) )
		else:
			file_paths.append( path )
	file_paths.sort()

	if len(file_paths) == 0:
		log.error("No partition files found to validate")
		sys.exit(1)
	log.info("Processing partitions for %d base topic models  ..." % len(file_paths) )

	# Load cached partitions
	all_partitions = []
	for file_path in file_paths:
		log.debug( "Loading partition from %s" % file_path )
		partition,cluster_doc_ids = unsupervised.util.load_partition( file_path )
		all_partitions.append( partition )

	r = len(all_partitions)
	log.info( "Evaluating stability of %d partitions with NMI ..." % r )
	# compute NMI of each pair of partitions	
	all_scores = []
	for i in range(r):
		for j in range(i+1,r):
			score = normalized_mutual_info_score( all_partitions[i], all_partitions[j] )
			all_scores.append( score )

	# Get overall score across all pairs
	all_scores = np.array( all_scores )
	tab = PrettyTable( ["statistic","stability"] )
	tab.align["statistic"] = "l"
	tab.add_row( [ "mean", "%.3f" % all_scores.mean() ] )
	tab.add_row( [ "median", "%.3f" %  np.median(all_scores) ] )
	tab.add_row( [ "sdev", "%.3f" % all_scores.std() ] )
	tab.add_row( [ "min", "%.3f" % all_scores.min() ] )
	tab.add_row( [ "max", "%.3f" % all_scores.max() ] )
	log.info( tab )
	
	# Write summary to CSV?
	if not options.out_path is None:
		log.info("Writing summary of results to %s" % options.out_path)
		unsupervised.util.write_table( options.out_path, tab )

	# Write histogram to CSV?
	if not options.hist_out_path is None:
		#bins = np.arange(0,1.1,0.1)
		bins = np.arange(0,1.01,0.05)
		inds = list(np.digitize(all_scores, bins, right=True))
		log.info("Writing histogram of results to %s" % options.hist_out_path)
		with open(options.hist_out_path,"w") as fout:
			fout.write("NMI,Count,Fraction\n")
			for ind, b in enumerate(bins):
				freq = inds.count(ind)
				frac = float(freq)/len(all_scores)
				fout.write("%.2f,%d,%.3f\n" % (b, freq, frac ) )

# --------------------------------------------------------------

if __name__ == "__main__":
	main()
