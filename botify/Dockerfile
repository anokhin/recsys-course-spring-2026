FROM python:3.11-slim

WORKDIR /app

COPY botify/requirements.txt /app/botify/requirements.txt
RUN pip install --no-cache-dir -r /app/botify/requirements.txt

COPY botify /app/botify
COPY script /app/script

RUN mkdir -p /app/botify/log

WORKDIR /app/botify
ENV PYTHONPATH="/app:/app/botify"
