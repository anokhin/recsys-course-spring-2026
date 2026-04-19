FROM python:3.11

WORKDIR /app

COPY requirements.txt .

RUN pip install -r requirements.txt

COPY botify ./botify

COPY ml_ranker_bundle.joblib /app/ml_ranker_bundle.joblib

COPY data ./data

RUN mkdir -p ./log

ENV PYTHONPATH "${PYTHONPATH}:/app"