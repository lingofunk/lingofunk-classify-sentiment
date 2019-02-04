import json
import os
import sys

import joblib
from lingofunk_classify_sentiment.model.utils import get_root, load_samples
from lingofunk_classify_sentiment.model.extract_reviews import save_reviews

from .preprocess import remove_stopwords_and_include_bigrams
from .train import train

ROOT = get_root()
CONFIG_PATH = os.path.join(ROOT, "config.json")

with open(CONFIG_PATH) as f:
    config = json.load(f)


def main(argv):
    if len(argv) != 2:
        programme_name = "lingofunk_classify_sentiment.model.naive_bayes.run"
        print(f"usage: PYTHONPATH=. python -m {programme_name} <category> <quantity>")
        sys.exit(2)
    category = argv[0]
    quantity = int(argv[1])

    try:
        (pos_words, neg_words) = load_samples(
            category, quantity, remove_stopwords_and_include_bigrams, save_reviews
        )
    except Exception:
        print("The data for this category and quantity are not found.")
        sys.exit(2)

    preprocessor_path = os.path.join(
        ROOT, config["models"]["naive_bayes"]["preprocessor"]
    )
    joblib.dump(remove_stopwords_and_include_bigrams, preprocessor_path, compress=0)

    print(f"Category: {category}")
    (accuracy, classifier, train_set, test_set) = train(pos_words, neg_words)
    classifier.show_most_informative_features()


if __name__ == "__main__":
    main(sys.argv[1:])
