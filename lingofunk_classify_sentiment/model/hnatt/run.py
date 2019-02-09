import os
import sys

import joblib
from lingofunk_classify_sentiment.config import config, fetch
from lingofunk_classify_sentiment.data.download_data import download_embedding
from lingofunk_classify_sentiment.data.load import (
    load_balanced_train_and_test_dataframes,
)
from lingofunk_classify_sentiment.data.extract_reviews import save_reviews
from lingofunk_classify_sentiment.model.hnatt.preprocess import normalize
from lingofunk_classify_sentiment.model.hnatt.scaffolding import HNATT


def main(argv):
    if len(argv) not in [2, 3]:
        programme_name = "lingofunk_classify_sentiment.model.hnatt.run"
        print(
            f"usage: PYTHONPATH=. python -m {programme_name} <category> <quantity>"
            " <embedding_name>"
        )
        sys.exit(2)
    category = argv[0]
    quantity = int(argv[1])

    embeddings_path = None
    if len(argv) == 3:
        embeddings_name = argv[2]
        embeddings_path = fetch(
            f'{config["embeddings"][embeddings_name]["basepath"]}.txt'
        )
        if not os.path.isfile(embeddings_path):
            download_embedding(embeddings_name)

    preprocessor_path = fetch(config["models"]["hnatt"]["preprocessor"])
    preprocessor_dir = os.path.dirname(preprocessor_path)
    if not os.path.exists(preprocessor_dir):
        os.makedirs(preprocessor_dir)

    joblib.dump(normalize, preprocessor_path, compress=0)

    (train_X, train_y), (test_X, test_y) = load_balanced_train_and_test_dataframes(
        category, quantity, normalize, save_reviews
    )

    # initialize HNATT
    h = HNATT()
    h.train(train_X, train_y, batch_size=256, epochs=16, embeddings_path=embeddings_path)

    h.load_weights()

    activation_maps = h.activation_maps(
        "they have some pretty interesting things here. i will definitely go back again."
    )
    print(activation_maps)


if __name__ == "__main__":
    main(sys.argv[1:])
