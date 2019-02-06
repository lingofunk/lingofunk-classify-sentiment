FROM python:3.6-jessie

COPY . /app
WORKDIR /app
ENV PYTHONPATH=.

RUN pip install --upgrade pip
RUN pip install pipenv
RUN pipenv install

CMD [\
    "pipenv", "run", "python", \
    "-m", "lingofunk_classify_sentiment.app.run", \
    "--port=8000", \
    "--model=hnatt"\
    ]



