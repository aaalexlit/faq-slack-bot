FROM prefecthq/prefect:2-python3.10

ENV EMBEDDING_CACHE_NAMESPACE=mlops_zoomcamp

RUN apt-get update && \
    apt-get install -y gcc python3-dev

RUN pip install -U pip

WORKDIR /usr/src

COPY ingest/requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY ingest/mlops/ingest_mlops.py ingest/mlops/
COPY ingest/readers ingest/readers
COPY ingest/utils ingest/utils
