#!/usr/bin/env python
"""
Tool to parse a collection of documents, where each file is stored in a separate plain text file.

Sample usage:
python parse-directory.py data/sample-text/ -o sample --tfidf --norm
"""
import os, os.path, sys, codecs, re, unicodedata
import logging as log
from optparse import OptionParser
import text.util

# --------------------------------------------------------------

def main():
	parser = OptionParser(usage="usage: %prog [options] dir1 dir2 ...")
	parser.add_option("-o", action="store", type="string", dest="prefix", help="output prefix for corpus files", default=None)
	parser.add_option("--df", action="store", type="int", dest="min_df", help="minimum number of documents for a term to appear", default=20)
	parser.add_option("--tfidf", action="store_true", dest="apply_tfidf", help="apply TF-IDF term weight to the document-term matrix")
	parser.add_option("--norm", action="store_true", dest="apply_norm", help="apply unit length normalization to the document-term matrix")
	parser.add_option("--minlen", action="store", type="int", dest="min_doc_length", help="minimum document length (in characters)", default=50)
	parser.add_option("-s", action="store", type="string", dest="stoplist_file", help="custom stopword file path", default=None)
	parser.add_option('-d','--debug',type="int",help="Level of log output; 0 is less, 5 is all", default=3)
	(options, args) = parser.parse_args()
	if( len(args) < 1 ):
		parser.error( "Must specify at least one directory" )	
	log.basicConfig(level=max(50 - (options.debug * 10), 10), format='%(asctime)-18s %(levelname)-10s %(message)s', datefmt='%d/%m/%Y %H:%M',)
	
	# Find all relevant files in directories specified by user
	filepaths = []
	args.sort()
	for in_path in args:
		if os.path.isdir( in_path ):
			log.info( "Searching %s for documents ..." % in_path )
			for fpath in text.util.find_documents( in_path ):
				filepaths.append( fpath )
		else:
			if in_path.startswith(".") or in_path.startswith("_"):
				continue
			filepaths.append( in_path )
	log.info( "Found %d documents to parse" % len(filepaths) )

	# Read the documents
	log.info( "Reading documents ..." )
	docs = []
	short_documents = 0
	doc_ids = []
	label_count = {}
	classes = {}
	for filepath in filepaths:
		# create the document ID
		label = os.path.basename( os.path.dirname( filepath ).replace(" ", "_") )
		doc_id = os.path.splitext( os.path.basename( filepath ) )[0]
		if not doc_id.startswith(label):
			doc_id = "%s_%s" % ( label, doc_id )
		# read body text
		log.debug( "Reading text from %s ..." % filepath )
		body = text.util.read_text( filepath )
		if len(body) < options.min_doc_length:
			short_documents += 1
			continue
		docs.append(body)	
		doc_ids.append(doc_id)	
		if label not in classes:
			classes[label] = set()
			label_count[label] = 0
		classes[label].add(doc_id)
		label_count[label] += 1
	log.info( "Kept %d documents. Skipped %d documents with length < %d" % ( len(docs), short_documents, options.min_doc_length ) )
	if len(classes) < 2:
		log.warning( "No ground truth available" )
		classes = None
	else:
		log.info( "Ground truth: %d classes - %s" % ( len(classes), label_count ) )

	# Convert the documents in TF-IDF vectors and filter stopwords
	if options.stoplist_file is None:
		stopwords = text.util.load_stopwords("text/stopwords.txt")
	elif options.stoplist_file.lower() == "none":
		log.info("Using no stopwords")
		stopwords = set()
	else:
		log.info( "Using custom stopwords from %s" % options.stoplist_file )
		stopwords = text.util.load_stopwords(options.stoplist_file)
	log.info( "Pre-processing data (%d stopwords, tfidf=%s, normalize=%s, min_df=%d) ..." % (len(stopwords), options.apply_tfidf, options.apply_norm, options.min_df) )
	(X,terms) = text.util.preprocess( docs, stopwords, min_df = options.min_df, apply_tfidf = options.apply_tfidf, apply_norm = options.apply_norm )
	log.info( "Built document-term matrix: %d documents, %d terms" % (X.shape[0], X.shape[1]) )
	
	# Store the corpus
	prefix = options.prefix
	if prefix is None:
		prefix = "corpus"
	log.info( "Saving corpus '%s'" % prefix )
	text.util.save_corpus( prefix, X, terms, doc_ids, classes )
  
# --------------------------------------------------------------

if __name__ == "__main__":
	main()
