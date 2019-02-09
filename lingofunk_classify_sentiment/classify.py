import os
import sys

import numpy as np
import tensorflow as tf

import joblib
from keras import backend as K
from keras.models import load_model
from lingofunk_classify_sentiment.config import config, fetch
from lingofunk_classify_sentiment.model.hnatt.scaffolding import HNATT


class Classifier:
    def __init__(self):
        model_config = config["models"]["current"]

        self.model_name = model_config["name"]

        print("Loading model...")
        weights_path = fetch(model_config["weights"])
        preprocessor_path = fetch(model_config["preprocessor"])

        extension = os.path.splitext(weights_path)[-1].lower()

        if extension == ".joblib":
            self.model = joblib.load(weights_path, mmap_mode="r")
        elif extension in (".h5", ".hdf5"):
            if self.model_name == "hnatt":
                tokenizer_path = fetch(model_config["tokenizer"])
                K.clear_session()
                h = HNATT()
                h.load_weights(weights_path, tokenizer_path)
                self.model = h
                self.graph = tf.get_default_graph()
            else:
                self.model = load_model(weights_path)
        print("Loading the preprocessing function...")
        self.preprocessor = joblib.load(preprocessor_path)

    def preprocess(self, text):
        return self.preprocessor(text)

    def classify(self, text):
        if self.model_name == "hnatt":
            with self.graph.as_default():
                return self.model.classify(self.preprocess(text))
        else:
            return self.model.classify(self.preprocess(text))

    def prob_classify(self, text):
        if self.model_name == "hnatt":
            with self.graph.as_default():
                return self.model.prob_classify(self.preprocess(text))
        else:
            dist = self.model.prob_classify(self.preprocess(text))
            return dist.prob(self.classify(text))

    def activation_maps(self, text):
        if self.model_name == "hnatt":
            with self.graph.as_default():
                return self.model.activation_maps(text, websafe=True)
        return []


def classify(argv):
    if len(argv) != 2:
        print(
            'usage: PYTHONPATH=. python -m lingofunk_classify_sentiment.classify <model_name> "<text>"'
        )
    model_name = argv[0]
    text = argv[1]

    classifier = Classifier()

    print(classifier.classify(text))


if __name__ == "__main__":
    classify(sys.argv[1:])
