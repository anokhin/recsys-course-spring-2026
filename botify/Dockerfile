FROM python:3.11

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY botify ./botify
COPY data ./data
COPY session_gate_ranker_bundle.joblib ./session_gate_ranker_bundle.joblib
RUN mkdir -p ./log

ENV PYTHONPATH "${PYTHONPATH}:/app"