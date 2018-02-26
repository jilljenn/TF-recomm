import argparse
from config import ARTICLE_FOLDER
from collections import defaultdict
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import dataio
import glob
import json
import os.path


parser = argparse.ArgumentParser(description='Generate tables and plots')
parser.add_argument('--dataset', type=str, nargs='?', default='fraction')
options = parser.parse_args()
DATASET_NAME = options.dataset
CSV_FOLDER, CSV_TRAIN, CSV_TEST, CSV_VAL, CONFIG, Q_NPZ = dataio.build_paths(DATASET_NAME)


if DATASET_NAME == 'assistments':
    categories = ['All models', 'wins + fails']
    categories_regexp = ['', 'wins, fails ']
elif DATASET_NAME == 'berkeley':
    categories = ['All models', 'd = 10']
    categories_regexp = ['', 'd = 10 ']
elif DATASET_NAME == 'timss2003':
    categories = ['users + items + skills', 'd = 5']
    categories_regexp = ['users, items, skills ', 'd = 5 ']
elif DATASET_NAME == 'fraction':
    categories = ['All models', 'users + items', 'd = 0']
    categories_regexp = ['', 'users, items ', 'd = 0 ']
else:
    categories = ['users + items + skills', 'd = 5']
    categories_regexp = ['users, items, skills', 'd = 5 ']

experiments = glob.glob('%s/*/results.json' % CSV_FOLDER)
fig, axes = plt.subplots(2, len(categories), figsize=(8, 12), sharex='col')  # x-axis will be shared across columns
acc, nll = axes

# Tables
labels = {}
TABLE_TEX = os.path.join(CSV_FOLDER, '{:s}-table.tex'.format(DATASET_NAME))
array = []
with open(TABLE_TEX, 'w') as latex:
    for filename in experiments:
        with open(filename) as f:
            results = json.load(f)
        short_legend, full_legend, latex_legend, active_agents = dataio.get_legend(results['args'])
        labels[filename] = full_legend
        line = '{:s} & '.format(latex_legend)
        line += ' & '.join('{:.4f}'.format(results['metrics'][metric]) for metric in ['ACC', 'AUC', 'NLL']) + r'\\'
        row = [latex_legend, results['args']['d']] + [results['metrics'][metric] for metric in ['ACC', 'AUC', 'NLL']]
        array.append(row)
        print(line)
        latex.write(line + '\n')
df = pd.DataFrame(array, columns=('model', 'dim', 'ACC', 'AUC', 'NLL')).sort_values('NLL').round(3)
df.to_latex(TABLE_TEX, column_format='c' * 5, escape=False, index=False)

# Curves
acc_curves = defaultdict(list)
nll_curves = defaultdict(list)
for filename in experiments:
    df = pd.read_csv(filename.replace('results.json', 'rlog.csv'))
    nb_epochs = len(df)
    for ax_id, category_regexp in enumerate(categories_regexp):
        if category_regexp in labels[filename] + ' ':
            curve, = acc[ax_id].plot(range(1, nb_epochs + 1), df['accuracy'], label=labels[filename])
            acc_curves[ax_id].append(curve)
            curve, = nll[ax_id].plot(range(1, nb_epochs + 1), df['ll_mcmc_all'], label=labels[filename])
            nll_curves[ax_id].append(curve)
for ax_id, category in enumerate(categories):
    acc[ax_id].legend(handles=acc_curves[ax_id])
    acc[ax_id].set_title(category)
    nll[ax_id].legend(handles=nll_curves[ax_id])
    nll[ax_id].set_xlabel('Epochs')
acc[0].set_ylabel('Accuracy')
nll[0].set_ylabel('Log-loss')
RESULTS_PDF = os.path.join(CSV_FOLDER, '{:s}-results.pdf'.format(DATASET_NAME))
plt.savefig(RESULTS_PDF, format='pdf')
#os.system('open {:s}'.format(TABLE_TEX))
os.system('cp {:s} {:s}tables/'.format(TABLE_TEX, ARTICLE_FOLDER))
os.system('cp {:s} {:s}figures/'.format(RESULTS_PDF, ARTICLE_FOLDER))
# plt.show()
