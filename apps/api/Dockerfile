FROM python:3.12-slim

WORKDIR /app

# system deps (optional but useful)
RUN apt-get update && apt-get install -y --no-install-recommends \
  build-essential curl \
  && rm -rf /var/lib/apt/lists/*

COPY apps/api/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY apps/api/ .
COPY core/ core/

EXPOSE 8000