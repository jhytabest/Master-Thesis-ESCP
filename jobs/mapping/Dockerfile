FROM python:3.11-slim

WORKDIR /app

RUN pip install --no-cache-dir pandas>=2.2.0 boto3>=1.34.0 google-genai>=0.6.0

COPY src /app/src

ENV PYTHONPATH=/app/src

ENTRYPOINT ["python", "-m", "mapping.propose"]
