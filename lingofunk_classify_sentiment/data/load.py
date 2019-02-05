import json
import os
from string import Template

import numpy as np
from tqdm import tqdm

import pandas as pd

from lingofunk_classify_sentiment.config import config, fetch

tqdm.pandas()
sample_template_filename = Template(fetch(config["datasets"]["yelp"]["sample_format"]))


def load_samples(category, quantity, preprocess, save=False):
    pos_reviews_fn = sample_template_filename.substitute(
        category=category.lower(), quantity=quantity, label="pos"
    )
    neg_reviews_fn = sample_template_filename.substitute(
        category=category.lower(), quantity=quantity, label="neg"
    )

    bothExist = os.path.isfile(pos_reviews_fn) and os.path.isfile(neg_reviews_fn)

    if not bothExist and save:
        save(category, quantity)

    pos_reviews = open(pos_reviews_fn, "r")
    neg_reviews = open(neg_reviews_fn, "r")

    pos_samples = [
        (preprocess((json.loads(review)["text"])), "pos") for review in pos_reviews
    ]
    neg_samples = [
        (preprocess((json.loads(review)["text"])), "neg") for review in neg_reviews
    ]

    return (pos_samples, neg_samples)


def polarize(rating):
    if rating > 3:
        return 2
    else:
        return 1


def to_one_hot(labels, dim=2):
    results = np.zeros((len(labels), dim))
    for i, label in enumerate(labels):
        results[i][label - 1] = 1
    return results


def get_text_and_sentiment(filename, preprocess):
    with open(filename, "r") as f:
        data = f.readlines()
    data = map(lambda x: x.rstrip(), data)
    data_json_str = "[" + ",".join(data) + "]"
    df = pd.read_json(data_json_str)[["stars", "text"]]
    text = df["text"].progress_apply(lambda x: preprocess(x))
    sentiment = df["stars"].apply(lambda x: polarize(x))
    return text, to_one_hot(sentiment)


def load_balanced_train_and_test_dataframes(
    category, quantity, preprocess, save=None, train_ratio=0.8
):
    print("Loading Yelp reviews...")
    pos_reviews_fn = sample_template_filename.substitute(
        category=category.lower(), quantity=quantity, label="pos"
    )
    neg_reviews_fn = sample_template_filename.substitute(
        category=category.lower(), quantity=quantity, label="neg"
    )

    bothExist = os.path.isfile(pos_reviews_fn) and os.path.isfile(neg_reviews_fn)

    if not bothExist and save:
        save(category, quantity)

    X_pos, y_pos = get_text_and_sentiment(pos_reviews_fn, preprocess)
    X_neg, y_neg = get_text_and_sentiment(neg_reviews_fn, preprocess)

    num_reviews = min(len(X_pos), len(X_neg))
    train_num = int(round(train_ratio * num_reviews))
    train_X = np.concatenate((X_neg[:train_num], X_pos[:train_num]), axis=0)
    train_y = np.concatenate((y_neg[:train_num], y_pos[:train_num]), axis=0)
    test_X = np.concatenate((X_neg[train_num:], X_pos[train_num:]), axis=0)
    test_y = np.concatenate((y_neg[train_num:], y_pos[train_num:]), axis=0)

    return (train_X, train_y), (test_X, test_y)


def load_glove_embedding(path, dim, word_index):
    embeddings_index = {}
    with open(path) as f:
        print("Generating GloVe embedding...")
        for line in tqdm(f):
            values = line.split()
            word = ''.join(values[:-dim])
            coefs = np.asarray(values[-dim:], dtype='float32')
            embeddings_index[word] = coefs

    embedding_matrix = np.zeros((len(word_index) + 1, dim))

    for word, i in word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector
    print("Loaded GloVe embedding.")

    return embedding_matrix
