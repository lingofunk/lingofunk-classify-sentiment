import json
import os
from pathlib import Path
from string import Template

def get_root():
    """Return project root folder"""
    return Path(__file__).parent.parent


ROOT = get_root()
CONFIG_PATH = os.path.join(ROOT, "config.json")

with open(CONFIG_PATH) as f:
    config = json.load(f)


sample_template_filename = Template(
    os.path.join(ROOT, config["datasets"]["yelp"]["sample_format"])
)


def load_samples(category, quantity, process_text, save=False):
    pos_reviews_fn = sample_template_filename.substitute(
        category=category.lower(), quantity=quantity, label="pos"
    )
    neg_reviews_fn = (
        sample_template_filename.substitute(
            category=category.lower(), quantity=quantity, label="neg"
        ),
    )

    bothExist = os.path.isfile(pos_reviews_fn) and os.path.isfile(neg_reviews_fn)

    if not bothExist and save:
        save(category, quantity)

    pos_reviews = open(pos_reviews_fn, "r")
    neg_reviews = open(neg_reviews_fn, "r")

    pos_samples = [
        (process_text((json.loads(review)["text"])), "pos") for review in pos_reviews
    ]
    neg_samples = [
        (process_text((json.loads(review)["text"])), "neg") for review in neg_reviews
    ]

    return (pos_samples, neg_samples)
