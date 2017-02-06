#!/usr/bin/env python
"""
Evaluate the stability (i.e. agreement) between a set of term rankings generated on the same dataset, using Average Term Stability (ATS).

Sample usage:
python eval-term-stability.py models/base/ranks*.pkl 
python eval-term-stability.py models/base/
"""
import os, sys
import logging as log
from optparse import OptionParser
import numpy as np
from prettytable import PrettyTable
import unsupervised.nmf, unsupervised.rankings, unsupervised.util

# --------------------------------------------------------------

def main():
	parser = OptionParser(usage="usage: %prog [options] rank_file1|directory1 ...")
	parser.add_option("-t", "--top", action="store", type="int", dest="top", help="number of top terms to use", default=10)
	parser.add_option("-o","--output", action="store", type="string", dest="out_path", help="path for CSV output file", default=None)
	# Parse command line arguments
	(options, args) = parser.parse_args()
	if( len(args) < 1 ):
		parser.error( "Must specify one or more term rankings or directories" )
	log.basicConfig(level=20, format='%(message)s')

	# Get list of all specified term ranking files
	file_paths = []
	for path in args:
		if not os.path.exists( path ):
			log.error("No such file or directory: %s" % path )
			sys.exit(1)
		if os.path.isdir(path):
			log.info("Searching %s for term ranking files" % path )
			for dir_path, dirs, files in os.walk(path):
				for fname in files:
					if fname.startswith("ranks") and fname.endswith(".pkl"):
						file_paths.append( os.path.join( dir_path, fname ) )
		else:
			file_paths.append( path )
	file_paths.sort()
	if len(file_paths) == 0:
		log.error("No term ranking files found to validate")
		sys.exit(1)
	log.info( "Processing %d topic models ..." % len(file_paths) )

	# Load cached ranking sets
	all_term_rankings = []
	for rank_path in file_paths:
		log.debug( "Loading term ranking set from %s ..." % rank_path )
		(term_rankings,labels) = unsupervised.util.load_term_rankings( rank_path )
		log.debug( "Set has %d rankings covering %d terms" % ( len(term_rankings), unsupervised.rankings.term_rankings_size( term_rankings ) ) )
		# do we need to truncate the number of terms in the ranking?
		if options.top > 1:
			term_rankings = unsupervised.rankings.truncate_term_rankings( term_rankings, options.top )
			log.debug( "Truncated to %d -> set now has %d rankings covering %d terms" % ( options.top, len(term_rankings), unsupervised.rankings.term_rankings_size( term_rankings ) ) )
		all_term_rankings.append( term_rankings )

	r = len(all_term_rankings)
	metric = unsupervised.rankings.JaccardBinary()
	matcher = unsupervised.rankings.RankingSetAgreement( metric )	

	# Perform pairwise comparisons evaluation for all models
	log.info( "Evaluating stability %d base term rankings with %s and top %d terms ..." % (r, str(metric), options.top) )
	all_scores = []
	for i in range(r):
		for j in range(i+1,r):
			try:
				score = matcher.similarity( all_term_rankings[i], all_term_rankings[j] )
				all_scores.append( score )
			except Exception as e:
				log.warning("Error occurred comparing pair (%d,%d): %s" % ( i, j, str(e) ) )
	log.info("Compared %d pairs of term rankings" % len(all_scores) )

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
	
	# Write to CSV?
	if not options.out_path is None:
		log.info("Writing summary of results to %s" % options.out_path)
		unsupervised.util.write_table( options.out_path, tab )

# --------------------------------------------------------------

if __name__ == "__main__":
	main()
