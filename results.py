from collections import defaultdict
from config import ARTICLE_FOLDER
import pandas as pd
import os.path
import glob
import json


interesting = ['PFA', 'AFM', 'IRT', 'MIRTb10', 'KTM(uia0)', 'KTM(uis0)', 'KTM(uis10)', 'KTM(uiswfWF1)']
full = set()

array = defaultdict(dict)
experiments = glob.glob('*/*/*/results.json')
datasets = set()
for experiment in experiments:
    with open(experiment) as f:
        results = json.load(f)
    if 'dataset' not in results['args'] or results['args']['dataset'][-2:] != '42':  # X42
        continue
    dataset = os.path.basename(results['args']['dataset'])
    datasets.add(dataset)
    short_legend = results['legends']['short']
    full_legend = results['legends']['full']
    shortname = full_legend.split(':')[0] if ':' in full_legend else 'KTM({:s})'.format(short_legend)
    if shortname == 'MIRTb':
        shortname += str(results['args']['d'])
    if shortname in interesting:
        full.add(full_legend)
    array[dataset][shortname] = results['metrics']['AUC']
df = pd.DataFrame.from_dict(array).round(4)
for dataset in datasets:
    extremum = df[dataset].max()
    winners = df.query('abs({:s} - @extremum) <= 0.00011'.format(dataset))
    df.loc[winners.index, dataset] = winners[dataset].map(lambda entry: r'\textbf{{{:.4f}}}'.format(entry))
# df.transpose()[interesting].fillna('--').to_latex(os.path.join(ARTICLE_FOLDER, 'summary.tex'), escape=False)
print(df)

for legend in full:
    print(legend)
