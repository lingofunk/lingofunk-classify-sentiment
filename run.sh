#!/usr/bin/env bash

pipenv shell
PYTHONPATH=. python -m lingofunk_classify_sentiment.model.hnatt.run Restaurants 1000000 glove-840B-300d
