FROM prefecthq/prefect:2-python3.10

RUN apt-get update && \
    apt-get install -y gcc python3-dev

RUN pip install -U pip

WORKDIR /usr/src

COPY ingest/requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY ingest/ml/ingest_ml.py ingest/ml/
COPY ingest/readers ingest/readers
COPY ingest/utils ingest/utils
