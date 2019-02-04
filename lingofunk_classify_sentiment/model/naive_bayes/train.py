import json
import os

import numpy as np
import joblib

import nltk.classify.util
from nltk.classify import NaiveBayesClassifier
from nltk.metrics import scores
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split

from lingofunk_classify_sentiment.model.utils import get_root

ROOT = get_root()
CONFIG_PATH = os.path.join(ROOT, "config.json")

with open(CONFIG_PATH) as f:
    config = json.load(f)


def train(pos_samples, neg_samples):
    model_path = os.path.join(ROOT, config["models"]["naive_bayes"]["weights"])

    samples = np.array(pos_samples + neg_samples)

    train_samples, test_samples = train_test_split(
        samples, test_size=0.2, random_state=42
    )

    if os.path.isfile(model_path):
        classifier = joblib.load(model_path).train(train_samples)
    else:
        classifier = nltk.NaiveBayesClassifier.train(train_samples)


    accuracy = nltk.classify.util.accuracy(classifier, test_samples)
    print(f"Finished training. The accuracy is {accuracy}.")
    test_trained_classifier(classifier, test_samples)

    return (accuracy, classifier, train_samples, test_samples)


def test_model_with_cross_validation(model, samples, n_folds=2):
    labels = [label for (text, label) in samples]
    cv = StratifiedKFold(labels, n_folds=n_folds, shuffle=True)

    accuracy = 0.0

    for traincv, testcv in cv:
        train_samples = samples[traincv]
        test_samples = samples[testcv]
        classifier = model.train(train_samples)
        accuracy += nltk.classify.util.accuracy(classifier, test_samples)

    accuracy /= n_folds

    return (accuracy, classifier, train_samples, test_samples)


def test_trained_classifier(classifier, test_samples):
    """Prints precision/recall statistics of a NLTK classifier"""
    import collections

    refsets = collections.defaultdict(set)
    testsets = collections.defaultdict(set)

    for i, (sample, label) in enumerate(test_samples):
        refsets[label].add(i)
        observed = classifier.classify(sample)
        testsets[observed].add(i)

    print("pos precision:", scores.precision(refsets["pos"], testsets["pos"]))
    print("pos recall:", scores.recall(refsets["pos"], testsets["pos"]))
    print("pos F-measure:", scores.f_measure(refsets["pos"], testsets["pos"]))
    print("neg precision:", scores.precision(refsets["neg"], testsets["neg"]))
    print("neg recall:", scores.recall(refsets["neg"], testsets["neg"]))
    print("neg F-measure:", scores.f_measure(refsets["neg"], testsets["neg"]))
