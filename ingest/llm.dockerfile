FROM prefecthq/prefect:2-python3.10

RUN apt-get update && \
    apt-get install -y gcc python3-dev

RUN pip install -U pip

WORKDIR /usr/src

COPY ingest/requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

ENV EMBEDDING_CACHE_NAMESPACE=llm_zoomcamp

COPY ingest/llm/ingest_llm.py ingest/llm/
COPY ingest/readers ingest/readers
COPY ingest/utils ingest/utils
