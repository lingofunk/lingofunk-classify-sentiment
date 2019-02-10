#!/usr/bin/env bash

LEARNING_RATE_START=0.0005
LEARNING_RATE_END=0.0015
LEARNING_RATE_STEP=0.0005

INPUT_SIZE_START=33
INPUT_SIZE_END=99
INPUT_SIZE_STEP=33

for LEARNING_RATE in $(seq $LEARNING_RATE_START $LEARNING_RATE_STEP $LEARNING_RATE_END); do
    for INPUT_SIZE in $(seq $INPUT_SIZE_START $INPUT_SIZE_STEP $INPUT_SIZE_END); do
        CMD="lingofunk_classify_sentiment.model.hnatt.run"
        VARS="Restaurants $1 glove-840B-300d $INPUT_SIZE $LEARNING_RATE"
        TRAIN_CMD="PYTHONPATH=. python -m $CMD $VARS"
        echo $TRAIN_CMD
        pipenv run bash -c "$TRAIN_CMD"
    done
done
