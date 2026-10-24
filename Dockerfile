FROM python:3.12-slim-bookworm
WORKDIR /app
ENV PYTHONPATH=/app/01_data:/app/02_model:/app/03_app:/app:/app/04_tests
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*
COPY requirements.txt ./requirements.txt
COPY Dockerfile ./Dockerfile
RUN pip install --no-cache-dir -r requirements.txt
COPY 01_data/ ./01_data/
COPY 02_model/ ./02_model/
COPY 03_app/ ./03_app/
COPY 04_tests/ ./04_tests/
CMD ["python", "-m", "daily.daily_batch", "--help"]
