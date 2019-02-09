#!/usr/bin/env bash

pipenv run bash -c "PYTHONPATH=. python -m lingofunk_classify_sentiment.model.hnatt.run Restaurants $1 glove-840B-300d"
