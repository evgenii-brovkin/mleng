import sys
import os
import pickle as pkl
import numpy as np
import pandas as pd
import yaml
from sklearn.model_selection import train_test_split


def split_data(features, targets, test_size, seed):
    with open(features, 'rb') as fd:
        features = pkl.load(fd)
    targets_df = pd.read_csv(targets)
    x_train, x_test, y_train, y_test = train_test_split(features, targets_df.died.values, test_size=test_size, random_state=seed)
    os.makedirs(os.path.join('data', 'train'), exist_ok=True)
    os.makedirs(os.path.join('data', 'test'), exist_ok=True)
    with open('data/train/train_pairs.pkl', 'wb') as f:
        pkl.dump((x_train, y_train), f)
    with open('data/test/test_pairs.pkl', 'wb') as f:
        pkl.dump((x_test, y_test), f)


if __name__ == '__main__':
    if len(sys.argv) != 3:
        sys.stderr.write('Arguments error. Usage:\n')
        sys.stderr.write('\tpython train_test_split.py features parsed\n')
        sys.exit(1)

    params = yaml.safe_load(open('params.yaml'))['train_test_split']
    feat_input = sys.argv[1]
    target_input = sys.argv[2]
    seed = params['seed']
    test_size = params['test_size']
    split_data(feat_input, target_input, test_size, seed)