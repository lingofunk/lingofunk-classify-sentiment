import json
import os
import sys

import joblib
from lingofunk_classify_sentiment.model.utils import get_root

ROOT = get_root()
CONFIG_PATH = os.path.join(ROOT, "config.json")

with open(CONFIG_PATH) as f:
    config = json.load(f)


class Classifier:
    def __init__(self, model_name):
        self.model_name = model_name
        weights_path = os.path.join(ROOT, config["models"][model_name]["weights"])
        preprocessor_path = os.path.join(
            ROOT, config["models"][model_name]["preprocessor"]
        )

        self.model = joblib.load(weights_path)
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
