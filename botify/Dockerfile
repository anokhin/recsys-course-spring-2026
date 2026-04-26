FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .

RUN pip install -r requirements.txt

COPY botify ./botify
COPY data ./data

RUN mkdir -p ./log

ENV PYTHONPATH "${PYTHONPATH}:/app"