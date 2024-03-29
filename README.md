lingofunk-classify-sentiment
============================

Yelp Review Sentiment Classifier

Usage
-----

#### Train a naive Bayes classifier

Pass the business type and the number of reviews required.

For instance, to train on 1000 reviews on restaurants, execute the following:

```shell
PYTHONPATH=. python -m lingofunk_classify_sentiment.model.naive_bayes.run Restaurants 1000
```

#### Train a hierarchical network with attention architecture (HNATT)

Pick the business type, the number of reviews required and one of the embedding files listed in config.json (for the naive embedding, leave the third argument blank).

To train on 1000 reviews on restaurants, execute the following:

```shell
PYTHONPATH=. python -m lingofunk_classify_sentiment.model.hnatt.run Restaurants 1000 glove-840B-300d
```

#### Classify a random review

Available models:

  - `naive_bayes`: a naive Bayes classifier, based on `NaiveBayesClassifier` from `nltk`.

```shell
PYTHONPATH=. python -m lingofunk_classify_sentiment.classify naive_bayes "Hello, world!"
```

| Metrics       |                     |
| ------------- | ------------------- |
| accuracy      | 0.7320675105485233. |
| pos precision | 0.9956709956709957  |
| pos recall    | 0.6460674157303371  |
| pos F-measure | 0.7836456558773425  |
| neg precision | 0.48148148148148145 |
| neg recall    | 0.9915254237288136  |
| neg F-measure | 0.6481994459833794  |

| Most Informative Features    |                               |
| ---------------------------- | ----------------------------- |
| horrible = True              | neg : pos    =     53.4 : 1.0 |
| ('food', 'ok') = True        | neg : pos    =     37.5 : 1.0 |
| ('awful', '.') = True        | neg : pos    =     33.4 : 1.0 |
| worst = True                 | neg : pos    =     33.0 : 1.0 |
| ('food', 'okay') = True      | neg : pos    =     31.4 : 1.0 |
| ('gross', '.') = True        | neg : pos    =     29.4 : 1.0 |
| ('terrible', '.') = True     | neg : pos    =     28.6 : 1.0 |
| ('food', 'poisoning') = True | neg : pos    =     27.3 : 1.0 |
| poisoning = True             | neg : pos    =     27.3 : 1.0 |
| ('mediocre', '.') = True     | neg : pos    =     27.3 : 1.0 |

  - `hnatt`: a hierarchical network with attention architecture naive Bayes classifier, based on minqi's [implementation](https://github.com/minqi/hnatt).

```shell
PYTHONPATH=. python -m lingofunk_classify_sentiment.classify hnatt "Hello, world!"
```


Installation
------------

### Requirements
- [pyenv](https://github.com/pyenv/pyenv)
- pipenv

```shell
pip install pipenv
```
### Setup

```shell
pyenv local $(cat .python-version)
pipenv install --dev
pipenv shell
```

### Utilities
#### Extracting reviews from the Yelp dataset

Pass the business type and the number of reviews required.

For instance, to generate 1000 reviews on restaurants, execute the following:

```shell
PYTHONPATH=. python -m lingofunk_classify_sentiment.data.extract_reviews Restaurants 1000
```

Licence
-------

MIT

Resources
---------

  1. Minqi's HNAtt [Implementation](https://github.com/minqi/hnatt)
