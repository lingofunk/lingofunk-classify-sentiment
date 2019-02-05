import sys

import joblib
from lingofunk_classify_sentiment.config import config, fetch


class Classifier:
    def __init__(self, model_name):
        self.model_name = model_name
        weights_path = fetch(config["models"][model_name]["weights"])
        preprocessor_path = fetch(config["models"][model_name]["preprocessor"])

        print("Loading model...")
        self.model = joblib.load(weights_path, mmap_mode="r")
        print("Loading the preprocessing function...")
        self.preprocess = joblib.load(preprocessor_path)

    def classify(self, text):
        return self.model.classify(self.preprocess(text))


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
