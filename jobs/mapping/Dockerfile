FROM python:3.11-slim

WORKDIR /app

RUN pip install --no-cache-dir pandas>=2.2.0

COPY src /app/src

ENV PYTHONPATH=/app/src

ENTRYPOINT ["python", "-m", "mapping.propose"]
