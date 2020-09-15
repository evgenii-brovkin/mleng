import sys
import os
import yaml
import pandas as pd


def filter_data(input_file, output_filtered, filter_date='2013-04-01'):
    df = pd.read_csv(input_file, encoding='cp1251', parse_dates=['date'])
    df = df.query("date < @filter_date")
    
    df.to_csv(output_filtered)


if __name__ == '__main__':
    params = yaml.safe_load(open('params.yaml'))['filter']

    if len(sys.argv) != 2:
        sys.stderr.write("Arguments error. Usage:\n")
        sys.stderr.write("\tpython filter.py data-file\n")
        sys.exit(1)

    date = params['date']

    input = sys.argv[1]
    output = os.path.join('data', 'filtered', 'filtered.csv')
    
    os.makedirs(os.path.join('data', 'filtered'), exist_ok=True)
    filter_data(input, output, filter_date=date)
