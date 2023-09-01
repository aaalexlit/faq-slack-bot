FROM python:3.10-slim-bookworm

RUN apt-get update && \
    apt-get install -y gcc

WORKDIR /usr/src/app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY main.py ./

CMD [ "python", "./main.py" ]