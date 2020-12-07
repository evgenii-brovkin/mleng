FROM python:3.8.6-slim-buster

COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt && rm requirements.txt

ARG USER_ID=1000
RUN useradd -M --no-log-init --system  --uid ${USER_ID} mlflow

USER mlflow
WORKDIR /app

COPY scripts .
CMD [ "python", "/app/test_mlflow_tracking.py"]
