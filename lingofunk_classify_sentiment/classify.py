import os
import sys

import numpy as np

import joblib
from keras import backend as K
import tensorflow as tf
from lingofunk_classify_sentiment.config import config, fetch
from lingofunk_classify_sentiment.model.hnatt.scaffolding import HNATT


class Classifier:
    def __init__(self, model_name):
        self.model_name = model_name
        weights_path = fetch(config["models"][model_name]["weights"])
        preprocessor_path = fetch(config["models"][model_name]["preprocessor"])

        print("Loading model...")
        extension = os.path.splitext(weights_path)[-1].lower()
        if extension == ".joblib":
            self.model = joblib.load(weights_path, mmap_mode="r")
        elif extension in (".h5", ".hdf5"):
            if model_name == "hnatt":
                K.clear_session()
                h = HNATT()
                h.load_weights(weights_path)
                self.model = h
                self.graph = tf.get_default_graph()
            else:
                self.model = load_model(weights_path)
        print("Loading the preprocessing function...")
        self.preprocess = joblib.load(preprocessor_path)

    def preprocess(self, text):
        return self.preprocess(text)

    def classify(self, text):
        with self.graph.as_default():
            return self.model.classify(self.preprocess(text))

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

    classifier = Classifier(model_name)

    print(classifier.classify(text))


if __name__ == "__main__":
    classify(sys.argv[1:])
