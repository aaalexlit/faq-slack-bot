FROM python:3.10-slim

# Install build tools, then clean up
RUN apt-get update \
 && apt-get install -y --no-install-recommends gcc python3-dev \
 && apt-get clean \
 && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir -U pip

WORKDIR /usr/src

# Install Python requirements
COPY ingest/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy ALL ingestion source code
COPY ingest ingest

# ---
# We’ll always override the CMD from the workflow, so keep a lightweight default
CMD ["python", "-c", "print('Specify the component when you run the container, e.g. `docker run … python -m ingest.ml.ingest_ml`')"]