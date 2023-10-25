FROM prefecthq/prefect:2-python3.10

RUN apt-get update && \
    apt-get install -y gcc  python3-dev

RUN pip install -U pip
RUN pip install pipenv

WORKDIR /usr/src/app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY *.py ./

ENTRYPOINT [ "python", "-u", "ingest_ml.py" ]