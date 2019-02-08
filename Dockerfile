FROM python:3.6-jessie

COPY . /app
COPY lingofunk_classify_sentiment/assets/model/hnatt/ /app/lingofunk_classify_sentiment/assets/model/hnatt/
WORKDIR /app
ENV PYTHONPATH=.

RUN pip install --upgrade pip
RUN pip install -r ./requirements.txt