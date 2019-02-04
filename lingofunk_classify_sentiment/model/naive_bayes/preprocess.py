import itertools
import re

import nltk
from nltk.collocations import BigramCollocationFinder
from nltk.corpus import stopwords
from nltk.metrics import BigramAssocMeasures
from nltk.tokenize import word_tokenize

nltk.download("stopwords")


def tokenize(text):
    """Splits a text to words, separates by space and punctuation,
    converts to lowercase."""
    return map(lambda token: token.lower(), re.findall(r"[\w']+|[.,!?;-]", text))


def remove_stopwords_and_include_bigrams(
    text, score_fn=BigramAssocMeasures.chi_sq, n_bigrams=500
):
    stopset = set(stopwords.words("english"))
    words = [word for word in tokenize(text) if word not in stopset]
    bigram_finder = BigramCollocationFinder.from_words(words)
    bigrams = bigram_finder.nbest(score_fn, n_bigrams)
    return dict([(ngram, True) for ngram in itertools.chain(words, bigrams)])
