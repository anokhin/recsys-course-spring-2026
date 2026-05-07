FROM python:3.7-slim-buster

WORKDIR /app

RUN sed -i 's|http://deb.debian.org/debian|http://archive.debian.org/debian|g' /etc/apt/sources.list \
    && sed -i 's|http://security.debian.org/debian-security|http://archive.debian.org/debian-security|g' /etc/apt/sources.list \
    && sed -i '/buster-updates/d' /etc/apt/sources.list \
    && apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install -r requirements.txt

COPY botify ./botify
COPY data ./data
COPY logistic_reranker_model.json ./logistic_reranker_model.json

RUN mkdir -p ./log

ENV PYTHONPATH "${PYTHONPATH}:/app"