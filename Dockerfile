FROM python:3.6-jessie

RUN pip install --upgrade pip
RUN pip install pipenv
RUN pipenv install

COPY . /app
WORKDIR /app

CMD [\
    "python", \
    "-m", "lingofunk_classify_sentiment.app.run", \
    "--port=8005", \
    "--model=hnatt"\
    ]



