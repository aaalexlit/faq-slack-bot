FROM python:3.10-slim

RUN apt-get update && \
    apt-get install -y gcc

WORKDIR /usr/src/app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir protobuf==3.20.2
RUN pip install --no-cache-dir pymilvus==2.3.0

COPY main.py ./

CMD [ "python", "-u", "./main.py" ]