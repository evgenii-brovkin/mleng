import os
import sys
import pandas as pd
import yaml
import pickle as pkl

from sklearn.feature_extraction.text import TfidfVectorizer


def extract_text_features(input_data_path, output, max_features, ngrams):
    df = pd.read_csv(input_data_path, encoding='cp1251')
    tfidf = TfidfVectorizer(ngram_range=(1, ngrams), max_features=max_features)
    features = tfidf.fit_transform(df.report)
    with open(output, 'wb') as f:
        pkl.dump(features, f)


if __name__ == '__main__':
    params = yaml.safe_load(open('params.yaml'))['feature_extraction']
    if len(sys.argv) != 2:
        sys.stderr.write('Arguments error. Usage:\n')
        sys.stderr.write(
            '\tpython featurization.py text-data-path \n'
        )
        sys.exit(1)

    text_input = sys.argv[1]
    output = os.path.join('features', 'features.pkl')

    max_features = params['max_features']
    ngrams = params['ngrams']
    
    os.makedirs(os.path.join('data', 'filtered'), exist_ok=True)
    extract_text_features(input, output, max_features=max_features, ngrams=ngrams)