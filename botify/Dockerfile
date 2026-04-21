FROM python:3.11-slim-bookworm

WORKDIR /app

COPY requirements.txt .

RUN pip install -r requirements.txt

COPY botify ./botify
COPY data ./data

RUN mkdir -p ./log

ENV PYTHONPATH "${PYTHONPATH}:/app"