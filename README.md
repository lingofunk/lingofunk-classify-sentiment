lingofunk-classify-sentiment
============================

![image](https://img.shields.io/pypi/v/lingofunk-classify-sentiment.svg%0A%20:target:%20https://pypi.python.org/pypi/lingofunk-classify-sentiment%0A%20:alt:%20Latest%20PyPI%20version)

[![Latest Travis CI build status](-.png)](-)

Yelp Review Sentiment Classifier

Usage
-----

Installation
------------

### Requirements

### Setup

#### Train a naive Bayes classifier

Pass the business type and the number of reviews required.

For instance, to train on 1000 reviews on restaurants, execute the following:

```shell
PYTHONPATH=. python -m lingofunk_classify_sentiment.model.naive_bayes.run Restaurants 1000
```

#### Classify a random review

Available models:

  - `naive_bayes`: a naive Bayes classifier, based on `NaiveBayesClassifier` from `nltk`.

```shell
PYTHONPATH=. python -m lingofunk_classify_sentiment.classify naive_bayes "Hello, world!"
```

### Utilities
#### Extracting reviews from the Yelp dataset

Pass the business type and the number of reviews required.

For instance, to generate 1000 reviews on restaurants, execute the following:

```shell
PYTHONPATH=. python -m lingofunk_classify_sentiment.model.extract_reviews Restaurants 1000
```


Compatibility
-------------

Licence
-------

Authors
-------