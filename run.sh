#!/usr/bin/env bash

pipenv shell
PYTHONPATH=. python -m lingofunk_classify_sentiment.model.hnatt.run Restaurants 100000 glove-840B-300d
