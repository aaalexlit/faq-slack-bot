FROM python:3.10-buster

WORKDIR /usr/src/app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY faiss_index faiss_index
COPY main.py ./

CMD [ "python", "./main.py" ]