# based on https://github.com/sfotiadis/yenlp/blob/master/extract_reviews.py

import json
import os
import sys
from string import Template

from lingofunk_classify_sentiment.data.load import get_root

ROOT = get_root()
CONFIG_PATH = os.path.join(ROOT, "config.json")

with open(CONFIG_PATH) as f:
    config = json.load(f)


business_data_filename = os.path.join(ROOT, config["datasets"]["yelp"]["ids"])
reviews_data_filename = os.path.join(ROOT, config["datasets"]["yelp"]["reviews"])
sample_template_filename = Template(
    os.path.join(ROOT, config["datasets"]["yelp"]["sample_format"])
)


def get_business_ids(category):
    """Gets the business ids for the given category"""
    with open(business_data_filename) as businesses:
        business_ids = []
        for business in businesses:
            business = json.loads(business)
            if business["categories"] and category in business["categories"].split():
                business_ids.append(business["business_id"])
    return business_ids


def save_reviews(category, quantity):
    """Saves the given number of reviews of a specific category to two files,
    one for each class(pos/neg)."""
    pos_reviews_filename = sample_template_filename.substitute(
        category=category.lower(), quantity=quantity, label="pos"
    )
    neg_reviews_filename = sample_template_filename.substitute(
        category=category.lower(), quantity=quantity, label="neg"
    )
    if os.path.isfile(pos_reviews_filename) and os.path.isfile(neg_reviews_filename):
        return

    pos_reviews = open(pos_reviews_filename, "w")
    neg_reviews = open(neg_reviews_filename, "w")

    business_ids = get_business_ids(category)

    cnt_pos = 0
    cnt_neg = 0

    with open(reviews_data_filename) as reviews:
        for review in reviews:
            # stop when quantity is reached
            if cnt_pos >= quantity and cnt_neg >= quantity:
                return None
            review = json.loads(review)
            if review["business_id"] in business_ids:
                # discard 3 star ratings
                if int(review["stars"]) > 3 and cnt_pos < quantity:
                    json.dump(review, pos_reviews)
                    pos_reviews.write("\n")
                    cnt_pos += 1
                elif int(review["stars"]) < 3 and cnt_neg < quantity:
                    json.dump(review, neg_reviews)
                    neg_reviews.write("\n")
                    cnt_neg += 1


def main(argv):
    if len(argv) != 2:
        print("Please list the category label and quantity of reviews required.")
        sys.exit(2)

    category = argv[0]
    quantity = int(argv[1])

    # load data
    try:
        print(f"Creating files with {quantity} reviews of the '{category}' category")
        save_reviews(category, quantity)
    except Exception:
        print("Alas! Something went wrong.")
        sys.exit(2)


if __name__ == "__main__":
    main(sys.argv[1:])
