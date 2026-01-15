FROM python:3.10-slim

WORKDIR /usr/src/app

COPY requirements.txt ./

# Install Python packages (all should have pre-built wheels)
RUN pip install --no-cache-dir -r requirements.txt && \
    pip cache purge && \
    find /usr/local -type f -name '*.pyc' -delete && \
    find /usr/local -type d -name '__pycache__' -delete

COPY main.py ./

CMD [ "python", "-u", "./main.py" ]