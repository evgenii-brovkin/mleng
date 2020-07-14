# ML Eng exercises

This repository contains modified version of [one of the solutions][kaggle-notebook] from Tweet Sentiment Extraction [competition][competition] for the purpose of using it in ML Eng course exercises.

## Cometition

Competition is based on simple classification task, but instead of just predicting sentiment, kagglers must give the subsentence which gave such prediction. Mote info on the competition [page][competition].

## Docker

The pipeline for evaluation consist of docker container built on modified Linux image. Optionally, there are other Docker files available at the `dockerfiles` directory, e.g. lightweighted version based on Alpine Linux. For ease of usage the default Dockerfile (the one in the repository root) is based on official tensorflow docker image.

## Setup

Use one of available dockerfiles to build docker image from root of the repo:

```shell
docker build -t tsfm -f Dockerfile .
```

Run the container with (note that you must mount three folders: data, models, output)

```shell
docker run -it --rm --name tsfm-pipeline -v ${pwd}/data:/app/data:ro -v ${pwd}/models:/app/models:ro -v ${pwd}/output:/app/output tsfm
```

Optionally, check the correctness of build using simple transformers pipeline:

```shell
docker run -it --rm tsfm bash
python -c "from transformers import pipeline; print(pipeline('sentiment-analysis')('I hate you'))"
```

## Pipeline

The evaluation pipeline consist of:

- Data loading
- Data preprocessing for the model
- Model inference
- Data output

By default the inference script use same, previously mounted directories. However, you can provide any other via script arguments. See the `scripts/inference.py` for more info about usage.s

[competition]: [https://www.kaggle.com/c/tweet-sentiment-extraction]
[kaggle-notebook]: [https://www.kaggle.com/cdeotte/tensorflow-roberta-0-705]