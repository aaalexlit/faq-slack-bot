FROM python:3.10-slim

# Install required build tools and clean up afterwards
RUN apt-get update \
 && apt-get install -y --no-install-recommends gcc python3-dev \
 && apt-get clean \
 && rm -rf /var/lib/apt/lists/*

# Upgrade pip to the latest version
RUN pip install --no-cache-dir -U pip

# Set the working directory inside the container
WORKDIR /usr/src

# Copy dependency list and install Python requirements
COPY ingest/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY ingest/mlops/ingest_mlops.py ingest/mlops/
COPY ingest/readers ingest/readers
COPY ingest/utils ingest/utils

# Default command
CMD ["python", "-m", "ingest.mlops.ingest_mlops"]