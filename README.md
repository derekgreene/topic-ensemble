topic-ensemblemodels/ensemble
===============

Ensemble topic modeling with matrix factorization

### Dependencies
Tested with Python 2.7 and Python 3.5, and requiring the following packages, which are available via PIP:

* Required: [numpy >= 1.8.0](http://www.numpy.org/)
* Required: [scikit-learn >= 0.14](http://scikit-learn.org/stable/)
* Required for utility tools: [prettytable >= 0.7.2](https://code.google.com/p/prettytable/)

### Basic Usage
##### Step 1. 
Before applying topic modeling to a corpus, the first step is to pre-process the corpus and store it in a suitable format. The script 'parse-directory.py' can be used to parse a directory of plain text documents. Here, we parse all .txt files in the directory or sub-directories of 'data/sample-text'. 

	python parse-directory.py data/sample-text/ -o sample --tfidf --norm

The output will be sample.pkl, stored as a Joblib binary file. The identifiers of the documents in the dataset correspond to the original text input filenames.

##### Step 2. 
Next, we generate a set of "base" topic models, which represent the members of the ensemble. We provide two different ways to do this.

Firstly, we can generate a specified number of base topic models using NMF and random initialization (the "Basic Ensemble" approach). For instance, we can generate 20 models, each containing *k=4* topics, where each NMF run will execute for a maximum of 100 iterations. The models will be written to the directory 'models/base' as a number of separate Joblib files.models/base
	
	python generate-nmf.py sample.pkl -k 4 -r 20 --maxiters 100 -o models/base

Alternatively, we can use the "K-Fold" ensemble approach. For instance, to execute 5 repetitions of 10 folds, we run: 

	python generate-kfold.py sample.pkl -k 4 -r 5 -f 10 --maxiters 100 -o models/base

##### Step 3. 
The next step is to combine the base topic models using an ensemble approach, to produce a final ensemble model. Note that we specify all of the factor files from the base topic models to combine, along with the number of overall ensemble topics (here it is again *k=4*). The model will be written as a number of files to the directory 'models/ensemble'.

	python combine-nmf.py sample.pkl models/base/*factors*.pkl -k 4 -o models/ensemble

We can display the top 10 terms in the topic descriptors for the final ensemble results:

	python display-top-terms.py models/ensemble/ranks_ensemble_k04.pkl 


Similarly, we can display the identifiers of the top-ranked document for each topic:

	python display-top-documents.py models/ensemble/factors_ensemble_k04.pkl 


