FROM python:3.7-slim-buster

WORKDIR /app

COPY requirements.txt .

RUN pip install -r requirements.txt

COPY botify ./botify
COPY data ./data

COPY session_gate_rf_bundle.joblib .
RUN mkdir -p ./log

ENV PYTHONPATH "${PYTHONPATH}:/app"
CMD ["gunicorn", "--bind", "0.0.0.0:5001", "--workers", "2", "app:app"]
