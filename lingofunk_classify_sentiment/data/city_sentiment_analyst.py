import numpy as np
import pandas as pd


from lingofunk_classify_sentiment.classify import Classifier
from lingofunk_classify_sentiment.config import config, fetch

DATASET_CSV = fetch("city")


class CitySentimentAnalyst:
    def __init__(self, n_bins=5):
        self.restaurant_reviews = pd.read_csv(DATASET_CSV)

        self.restaurant_reviews = self.restaurant_reviews.groupby(["business_id"]).agg(
            {"business_id": tuple, "text": list}
        )

        self.id2rest = list(
            map(lambda x: x[0], self.restaurant_reviews["business_id"].values)
        )
        self.rest2id = dict()
        for i, rest in enumerate(self.id2rest):
            self.rest2id[rest] = i

        self.restaurant_reviews = self.restaurant_reviews["text"].values
        self.n_comments_total = sum(
            len(restaurant) for restaurant in self.restaurant_reviews
        )
        self.n_restaurants = len(self.restaurant_reviews)
        self.lens_restaurants = np.array(list(map(len, self.restaurant_reviews)))

        self.classifier = Classifier()

        self.n_bins = n_bins

        self.sentiments = self.__init_sentiments__()

    @classmethod
    def compute_sentiments(cls):
        city_sentiment_analyst = cls()
        return city_sentiment_analyst.__init_sentiments__()

    def calculate_bin(self, x):
        int_x = int(x)
        if int_x == self.n_bins:
            int_x -= 1
        return int_x

    def __init_sentiments__(self):
        sentiments = np.zeros(shape=(self.n_restaurants, self.n_bins))
        for i in range(self.n_restaurants):
            for review in self.restaurant_reviews[i]:
                prediction = self.classifier.classify(review)
                sentiments[i][self.calculate_bin(prediction)] += 1
        return sentiments

    def get_histogram_for_restaurant_id(self, i):
        return self.sentiments[i] / sum(self.sentiments[i])

    def get_histogram_for_restaurant_name(self, rest):
        return self.get_histogram_for_restaurant_id(self.rest2id[rest])


if __name__ == "__main__":
    CitySentimentAnalyst.compute_sentiments()
