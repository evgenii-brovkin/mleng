# ML Eng exercises

This repository contains a modified version of [one of the solutions][kaggle-notebook] from Tweet Sentiment Extraction [competition][competition] to use it in ML Eng course exercises.

## Competition

Competition is based on the simple classification task, but instead of just predicting sentiment, kagglers must give the subsentence which gave such prediction. More info on the competition [page][competition].

## Docker

The pipeline for evaluation consists of docker container built on a modified Linux image. Optionally, there are other Docker files available at the `dockerfiles` directory, e.g. lightweight version based on Alpine Linux. For ease of usage, the default Dockerfile (the one in the repository root) is based on the official TensorFlow docker image.

## Setup

Use one of the available dockerfiles to build docker image from the root of the repo:

```shell
docker build -t tsfm -f Dockerfile .
```

Run the container with (note that you must mount three folders: data, models, output)

```shell
docker run -it --rm --name tsfm-pipeline  
-v ${pwd}/data:/home/ds/app/data:ro  
-v ${pwd}/models:/home/ds/app/models:ro  
-v ${pwd}/output:/home/ds/app/output  
tsfm
```

Optionally, check the correctness of build using simple transformers pipeline:

```shell
docker run -it --rm tsfm bash
python -c "from transformers import pipeline; print(pipeline('sentiment-analysis')('I hate you'))"
```

## Pipeline

The evaluation pipeline consists of:

- Data loading
- Data preprocessing for the model
- Model inference
- Data output

By default, the inference script uses same, previously mounted directories. However, you can provide any other via script arguments. See the `scripts/inference.py` for more info about usage.

[competition]: [https://www.kaggle.com/c/tweet-sentiment-extraction]
[kaggle-notebook]: [https://www.kaggle.com/cdeotte/tensorflow-roberta-0-705]