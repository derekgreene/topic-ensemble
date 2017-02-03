topic-ensemble
===============

Ensemble topic modeling with matrix factorization

### Dependencies
Tested with Python 2.7 and Python 3.5, and requiring the following packages, which are available via PIP:

* Required: [numpy >= 1.8.0](http://www.numpy.org/)
* Required: [scikit-learn >= 0.14](http://scikit-learn.org/stable/)
* Required for utility tools: [prettytable >= 0.7.2](https://code.google.com/p/prettytable/)

### Basic Usage
##### Step 1. 
Before applying topic modeling to a corpus, the first step is to pre-process the corpus and store it in a suitable format. The script 'parse-directory.py' can be used to parse a directory of plain text documents. Here, we parse all .txt files in the directory or sub-directories of 'data/sample-text'. The output file will have the prefix 'sample'.

	python parse-directory.py data/sample-text/ -o sample --tfidf --norm

The output will be a number of Joblib binary files, with the main corpus file being named 'sample.pkl'.

##### Step 2. 
Next, we generate a set of "base" topic models, which represent the members of the ensemble. We provide two different ways to do this.

Firstly, we can generate a specified number of base topic models using NMF and random initialization (the "Basic Ensemble" approach). For instance, we can generate 20 models, each containing *k=4* topics, where each NMF run will execute for a maximum of 100 iterations. The models will be written to the directory 'base-nmf'.
	
	python generate-nmf.py sample.pkl -k 4 -r 20 --maxiters 100 -o base-nmf

Alternatively, we can use the "K-Fold" ensemble approach. For instance, to execute 10 repetitions of 10 folds, we run: 

	python generate-kfold.py sample.pkl -k 4 -r 10 -f 10 --maxiters 100 -o base-nmf

##### Step 3. 
The next step is to combine the base topic models using an ensemble approach, to produce a final ensemble model. Note that we specify all of the base topic models to combine and the number of overall ensemble topics (here it is again *k=4*).

	python combine-nmf.py sample.pkl base-nmf/*factors*.pkl -k 4 -o ensemble-nmf

We can display the top 10 terms for the final ensemble topics:

	python display-topics.py ensemble-nmf/ranks_ensemble_k06.pkl 

### Evaluation Measures

To apply various external validation measures to evaluate the accuracy of the partition resulting from a topic model (i.e. single membership assignments), relative to a set of ground truth classes.

	python eval-partition-accuracy.py sample.pkl base-nmf/*partition*.pkl 

To evaluate the stability (i.e. agreement) between a set of partitions generated on the same dataset:

	python eval-partition-stability.py base-nmf/*partition*.pkl 
	

