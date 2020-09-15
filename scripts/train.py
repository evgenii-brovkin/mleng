import sys
import os
import pickle
import numpy as np
import yaml
from sklearn.linear_model import LogisticRegressionCV


def train_model(train_pairs, model_dump, Cs, penalty, seed):
    with open(train_pairs, 'rb') as fd:
        x, y = pickle.load(fd)

    clf = LogisticRegressionCV(
        Cs=Cs,
        penalty=penalty,
        n_jobs=2,
        solver='saga',
        random_state=seed,
        l1_ratios=[0.2, 0.5, 0.8]
    )
    clf.fit(x, y)

    with open(model_dump, 'wb') as fd:
        pickle.dump(clf, fd)

if __name__ == '__main__':
    if len(sys.argv) != 3:
        sys.stderr.write('Arguments error. Usage:\n')
        sys.stderr.write('\tpython train.py train_pairs model\n')
        sys.exit(1)

    params = yaml.safe_load(open('params.yaml'))['train']
    input = sys.argv[1]
    output = sys.argv[2]
    Cs = params['Cs']
    penalty = params['penalty']
    seed = params['seed']
    os.makedirs('models', exist_ok=True)
    train_model(input, output, Cs, penalty, seed)

