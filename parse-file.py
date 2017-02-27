#!/usr/bin/env python
"""
Tool to parse a collection of documents, where each document is a single line in one or more text files.

Sample usage:
python parse-file.py data/sample.txt -o sample --tfidf --norm
"""
import os, os.path, sys, codecs, re, unicodedata
import logging as log
from optparse import OptionParser
import text.util

# --------------------------------------------------------------

def main():
	parser = OptionParser(usage="usage: %prog [options] file1 file2 ...")
	parser.add_option("-o", action="store", type="string", dest="prefix", help="output prefix for corpus files", default=None)
	parser.add_option("--df", action="store", type="int", dest="min_df", help="minimum number of documents for a term to appear", default=20)
	parser.add_option("--tfidf", action="store_true", dest="apply_tfidf", help="apply TF-IDF term weight to the document-term matrix")
	parser.add_option("--norm", action="store_true", dest="apply_norm", help="apply unit length normalization to the document-term matrix")
	parser.add_option("--minlen", action="store", type="int", dest="min_doc_length", help="minimum document length (in characters)", default=50)
	parser.add_option("-s", action="store", type="string", dest="stoplist_file", help="custom stopword file path", default=None)
	parser.add_option('-d','--debug',type="int",help="Level of log output; 0 is less, 5 is all", default=3)
	(options, args) = parser.parse_args()
	if( len(args) < 1 ):
		parser.error( "Must specify at least one input file" )	
	log.basicConfig(level=max(50 - (options.debug * 10), 10), format='%(message)s')

	# Read the documents
	docs = []
	short_documents = 0
	doc_ids = []
	for in_path in args:
		file_count = 0
		log.info( "Reading documents from %s, one per line ..." % in_path )
		fin = codecs.open(in_path, 'r', encoding="utf8", errors='ignore')
		for line in fin.readlines():
			body = line.strip()
			if len(body) < options.min_doc_length:
				short_documents += 1
				continue
			doc_id = "%05d" % ( len(doc_ids) + 1 )
			docs.append(body)	
			doc_ids.append(doc_id)	
			file_count += 1
		log.info( "Kept %d documents from %s" % (file_count, in_path) )
	log.info( "Kept %d documents. Skipped %d documents with length < %d" % ( len(docs), short_documents, options.min_doc_length ) )

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
	text.util.save_corpus( prefix, X, terms, doc_ids, None )
  
# --------------------------------------------------------------

if __name__ == "__main__":
	main()
