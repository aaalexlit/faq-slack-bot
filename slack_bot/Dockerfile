FROM python:3.10-slim AS compile-image

# RUN apt-get install -y --no-install-recommends build-essential gcc

RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc python3-dev && \
    apt-get remove -y gcc && apt-get -y autoremove


COPY requirements.txt .
RUN pip install --user -r requirements.txt
RUN pip install --user protobuf==3.20.2
RUN pip install --user pymilvus==2.3.0

FROM python:3.10-slim AS build-image

COPY --from=compile-image /root/.local /root/.local
# Make sure scripts in .local are usable:
ENV PATH=/root/.local/bin:$PATH

WORKDIR /usr/src/app

COPY main.py ./

CMD [ "python", "-u", "./main.py" ]