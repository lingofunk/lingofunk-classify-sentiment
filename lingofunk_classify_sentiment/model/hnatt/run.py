import os
import sys

import joblib
from lingofunk_classify_sentiment.config import config, fetch
from lingofunk_classify_sentiment.data.download_data import download_embedding
from lingofunk_classify_sentiment.data.load import (
    load_balanced_train_and_test_dataframes
)
from lingofunk_classify_sentiment.data.extract_reviews import save_reviews
from lingofunk_classify_sentiment.model.hnatt.preprocess import normalize
from lingofunk_classify_sentiment.model.hnatt.scaffolding import HNATT


def main(argv):
    if len(argv) != 2:
        programme_name = "lingofunk_classify_sentiment.model.hnatt.run"
        print(f"usage: PYTHONPATH=. python -m {programme_name} <category> <quantity>")
        sys.exit(2)
    category = argv[0]
    quantity = int(argv[1])

    try:
        (train_X, train_y), (test_X, test_y) = load_balanced_train_and_test_dataframes(
            category, quantity, normalize
        )
    except Exception:
        print("The data for this category and quantity have not been found.")
        sys.exit(2)

    # embeddings_name = "glove-840B-300d"
    # embeddings_path = fetch(f'{config["embeddings"][embeddings_name]["basepath"]}.txt')

    # if not os.path.isfile(embeddings_path):
    #     download_embedding(embeddings_name)

    # initialize HNATT
    h = HNATT()
    # h.train(train_X, train_y, batch_size=16, epochs=16, embeddings_path=embeddings_path)
    h.train(train_X, train_y, batch_size=16, epochs=16, embeddings_path=None)

    h.load_weights()

    # embeddings = h.word_embeddings(train_x)
    # preds = h.predict(train_x)
    # print(preds)
    # import pdb; pdb.set_trace()

    # print attention activation maps across sentences and words per sentence
    activation_maps = h.activation_maps(
        "they have some pretty interesting things here. i will definitely go back again."
    )
    print(activation_maps)

    preprocessor_path = fetch(config["models"]["hnatt"]["preprocessor"])
    joblib.dump(remove_stopwords_and_include_bigrams, preprocessor_path, compress=0)


if __name__ == "__main__":
    main(sys.argv[1:])
