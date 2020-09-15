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
docker run -it --rm --name tsfm-pipeline -v ${pwd}/data:/home/ds/app/data:ro -v ${pwd}/models:/home/ds/app/models:ro -v ${pwd}/output:/home/ds/app/output tsfm
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

## DVC

For DVC separate dataset was used - `data/data.csv`. Preprocessing pipeline for that data stored in `dvc.yaml` file. To reproduce results run `dvc repro`.

### DVC pipeline

The pipeline consist of several steps:

- Data filtering by date
- Splitting raw data into text and parsed parts
- Extracting text features from reports

Described states have been produced via the next commands:

#### Filtering:
`dvc run -n filter -p filter.date -d scripts/filter.py -d data/data.csv -o data/filtered python scripts/filter.py data/data.csv`
#### Split:
`dvc run -n split -d scripts/split.py -d data/filtered/filtered.csv -o data/splitted python scripts/split.py data/filtered/filtered.csv`
#### Feature extraction:
`dvc run -n feature_extraction -p feature_extraction.max_features,feature_extraction.ngrams -d scripts/featurization.py -d data/splitted/report.csv -o data/features python scripts/featurization.py data/splitted/report.csv`
#### Train/test split:
`dvc run -n train_test_split -p train_test_split.seed,train_test_split.test_size -d scripts/train_test_split.py -d data/splitted/parsed.csv -d data/features/features.pkl -o data/train -o data/test python scripts/train_test_split.py data/features/features.pkl data/splitted/parsed.csv`
#### Training:
`dvc run -n train -p train.seed,train.Cs,train.penalty -d scripts/train.py -d data/train/train_pairs.pkl -o models python scripts/train.py data/train/train_pairs.pkl models/LogRegElasticNet.pkl`
#### Evaluation:
`dvc run -n evaluate -d scripts/evaluate.py -d data/test/test_pairs.pkl -d models/LogRegElasticNet.pkl -M results/logreg_scores.json python scripts/evaluate.py models/LogRegElasticNet.pkl data/test/test_pairs.pkl results/logreg_scores.json`
