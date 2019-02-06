FROM python:3.6-jessie

COPY . /app
WORKDIR /app
ENV PYTHONPATH=.

RUN pip install --upgrade pip
RUN pip install -r ./requirements.txt

CMD [\
    "python", \
    "-m", "lingofunk_classify_sentiment.app.run", \
    "--port=8000", \
    "--model=hnatt"\
    ]



