import json
import sys

import joblib

with open("config.json") as f:
    config = json.load(f)


class Classifier:
    def __init__(self, model_name):
        self.model = joblib.load(config["models"][model_name]["name"])
        self.preprocess = joblib.load(config["models"][model_name]["preprocessor"])

    def classify(self, text):
        return self.model.classify(self.preprocess(text))


def classify(argv):
    if len(argv) != 2:
        print(
            "usage: PYTHONPATH=. python -m lingofunk_classify_sentiment.classify <model_name> <text>"
        )
    model_name = argv[0]
    text = argv[1]

    classifier = Classifier(model_name)

    return classifier.classify(text)


if __name__ == "__main__":
    main(sys.argv[1:])
