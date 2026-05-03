FROM python:3.7-slim-buster

WORKDIR /app

RUN sed -i 's|http://deb.debian.org/debian|http://archive.debian.org/debian|g' /etc/apt/sources.list && \
    sed -i 's|http://deb.debian.org/debian-security|http://archive.debian.org/debian-security|g' /etc/apt/sources.list && \
    sed -i '/buster-updates/d' /etc/apt/sources.list && \
    apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN python -m pip install --no-cache-dir \
    pip==21.3.1 setuptools==59.8.0 wheel==0.37.1 && \
    pip install --no-cache-dir --only-binary :all: -r requirements.txt

COPY botify ./botify
COPY data ./data

RUN mkdir -p ./log

ENV PYTHONPATH="${PYTHONPATH}:/app"