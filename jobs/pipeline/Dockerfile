FROM python:3.11-slim

WORKDIR /app

RUN pip install --no-cache-dir boto3>=1.34.0 pandas>=2.2.0 pyarrow>=14.0.0 scikit-learn>=1.4.0

COPY src /app/src

ENV PYTHONPATH=/app/src

ENTRYPOINT ["python", "-m", "pipeline.cli"]
