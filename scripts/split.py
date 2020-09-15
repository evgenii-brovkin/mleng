import sys
import os
import pandas as pd


def split_data(input_file, output_parsed, output_report):
    df = pd.read_csv(input_file, encoding='cp1251', parse_dates=['date'])
    
    parsed_df = df.loc[:, ["date","total","acc_injured","injured","died","drunk","no_license","overspeed","pedestrians","crosswalk"]]
    report_df = df.loc[:, ["date","report"]]
    
    parsed_df.to_csv(output_parsed, index=False)
    report_df.to_csv(output_report, encoding='cp1251', index=False)


if __name__ == '__main__':
    if len(sys.argv) != 2:
        sys.stderr.write("Arguments error. Usage:\n")
        sys.stderr.write("\tpython split.py data-file\n")
        sys.exit(1)

    input = sys.argv[1]
    output_parsed = os.path.join('data', 'splitted', 'parsed.csv')
    output_report = os.path.join('data', 'splitted', 'report.csv')

    os.makedirs(os.path.join('data', 'splitted'), exist_ok=True)
    split_data(input, output_parsed, output_report)
