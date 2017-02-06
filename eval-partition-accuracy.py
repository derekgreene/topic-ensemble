#!/usr/bin/env python
"""
Apply various external validation measures to evaluate the accuracy of the partition 
resulting from a topic model (i.e. single membership assignments), relative to a set of ground truth classes.

Sample usage:
python eval-partition-accuracy.py sample.pkl models/base/*partition*.pkl 
"""
import os, os.path, sys
import logging as log
from optparse import OptionParser
from prettytable import PrettyTable
import numpy as np
from sklearn.metrics.cluster import normalized_mutual_info_score, adjusted_mutual_info_score, adjusted_rand_score
import text.util, unsupervised.util

# --------------------------------------------------------------

def validate( measure, classes, clustering ):
	if measure == "nmi":
		return normalized_mutual_info_score( classes, clustering )
	elif measure == "ami":
		return adjusted_mutual_info_score( classes, clustering )
	elif measure == "ari":
		return adjusted_rand_score( classes, clustering )
	log.error("Unknown validation measure: %s" % measure )
	return None

# --------------------------------------------------------------

def main():
	parser = OptionParser(usage="usage: %prog [options] corpus_file partition_file1|directory1 ...")
	parser.add_option("-s", "--summmary", action="store_true", dest="summary", help="display summary results only")
	parser.add_option("-o","--output", action="store", type="string", dest="out_path", help="path for CSV output file", default=None)
	parser.add_option("-m", "--measures", action="store_true", dest="measures", help="comma-separated list of validation measures to use (default is nmi)", default="nmi" )
	# Parse command line arguments
	(options, args) = parser.parse_args()
	if( len(args) < 2 ):
		parser.error( "Must specify at least a corpus and one or more partitions/directories" )	
	log.basicConfig(level=20, format='%(message)s')
	measures = [ x.strip() for x in options.measures.lower().split(",") ]

	log.info ("Reading corpus from %s ..." % args[0] )
	# Load the cached corpus
	(X,terms,doc_ids,classes) = text.util.load_corpus( args[0] )
	if classes is None:
		log.error( "Error: No class information available for this corpus")
		sys.exit(1)

	# Convert a map to a list of class indices
	classes_partition = unsupervised.util.clustermap_to_partition( classes, doc_ids )

	# Get list of all specified partition files
	file_paths = []
	for path in args[1:]:
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

	header = ["model"]
	for measure in measures:
		header.append(measure)
	tab = PrettyTable(header)
	tab.align["model"] = "l"

	scores = {}
	for measure in measures:
		scores[measure] = []

	for file_path in file_paths:
		log.debug( "Loading partition from %s" % file_path )
		partition,cluster_doc_ids = unsupervised.util.load_partition( file_path )
		k = max(partition) + 1
		# does the number of documents match up?
		if len(doc_ids) != len(cluster_doc_ids):
			log.warning("Error: Cannot compare clusterings on different data")
			continue
		# perform validation
		row = [file_path]
		for measure in measures:
			score = validate(measure,classes_partition,partition)
			scores[measure].append(score)
			row.append( "%.3f" % score )
		if not options.summary:
			tab.add_row(row)

	# display an overall summary?
	if options.summary or len(file_paths) > 1:
		# add mean
		row = ["mean"]
		for measure in measures:
			# convert to a NP array
			scores[measure] = np.array(scores[measure])
			row.append( "%.3f" % scores[measure].mean() )
		tab.add_row(row)
		row = ["median"]
		for measure in measures:
			row.append( "%.3f" % np.median(scores[measure]) )
		tab.add_row(row)
		# add standard deviation
		row = ["sdev"]
		for measure in measures:
			row.append( "%.3f" % scores[measure].std() )
		tab.add_row(row)
		# add range
		row = ["min"]
		for measure in measures:
			row.append( "%.3f" % scores[measure].min() )
		tab.add_row(row)
		row = ["max"]
		for measure in measures:
			row.append( "%.3f" % scores[measure].max() )
		tab.add_row(row)
	log.info( tab )

	# Write to CSV?
	if not options.out_path is None:
		log.info("Writing summary of results to %s" % options.out_path)
		unsupervised.util.write_table( options.out_path, tab )

# --------------------------------------------------------------

if __name__ == "__main__":
	main()
