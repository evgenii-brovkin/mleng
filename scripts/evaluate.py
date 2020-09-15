import sys
import os
import pickle
import json

from sklearn.metrics import precision_recall_fscore_support
import sklearn.metrics as metrics


def eval_model(model_file, test_pairs_file, scores_file, plots_file=None):
    with open(model_file, 'rb') as fd:
        model = pickle.load(fd)
    with open(test_pairs_file, 'rb') as fd:
        x, y_true = pickle.load(fd)

    y_pred = model.predict(x)
    precision, recall, fscore, support = precision_recall_fscore_support(y_true, y_pred, average='macro')

    os.makedirs('results', exist_ok=True)
    with open(scores_file, 'w') as fd:
        json.dump({'precision': precision, 'recall': recall, 'f1': fscore, 'n_samples': y_pred.shape[0]}, fd)


if __name__ == '__main__':
    if len(sys.argv) != 4:
        sys.stderr.write('Arguments error. Usage:\n')
        sys.stderr.write('\tpython evaluate.py model test_pairs scores\n')
        sys.exit(1)

    model_file = sys.argv[1]
    test_file = sys.argv[2]
    scores_file = sys.argv[3]

    eval_model(model_file, test_file, scores_file)

