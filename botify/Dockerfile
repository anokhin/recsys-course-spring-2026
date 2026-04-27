FROM python:3.10-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY botify ./botify
COPY data ./data

ENV PYTHONPATH=/app

RUN python -m botify.ml.train \
        --data-dir /app/data \
        --output-dir /app/data/ml

RUN mkdir -p ./log
