FROM python:3.10-slim

RUN pip install --no-cache-dir -U pip

WORKDIR /usr/src

# Install Python requirements (all should have pre-built wheels)
COPY ingest/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt && \
    pip cache purge && \
    find /usr/local -type f -name '*.pyc' -delete && \
    find /usr/local -type d -name '__pycache__' -delete

# Copy ALL ingestion source code
COPY ingest ingest

# ---
# We'll always override the CMD from the workflow, so keep a lightweight default
CMD ["python", "-c", "print('Specify the component when you run the container, e.g. `docker run … python -m ingest.ml.ingest_ml`')"]