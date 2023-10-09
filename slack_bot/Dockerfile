FROM python:3.10-slim

RUN apt-get update && \
    apt-get install -y gcc python3-dev

WORKDIR /usr/src/app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY main.py ./

CMD [ "python", "-u", "./main.py" ]