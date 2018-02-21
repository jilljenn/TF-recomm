from config import *
from collections import defaultdict
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob
import json


categories = {
    'users, items, skills': 0,
    'd = 5 ': 1
}

experiments = glob.glob('%s/*/results.json' % CSV_FOLDER)
fig, axes = plt.subplots(2, len(categories), figsize=(8, 16), sharex='col')  # x-axis will be shared across columns
acc, nll = axes

# Tables
labels = {}
with open(os.path.join(CSV_FOLDER, '{:s}-table.tex'.format(DATASET_NAME)), 'w') as latex:
    for filename in experiments:
        with open(filename) as f:
            results = json.load(f)
        labels[filename] = results['legends']['full']
        line = '{:s} & '.format(results['legends']['latex'])
        line += ' & '.join('{:.4f}'.format(results['metrics'][metric]) for metric in ['ACC', 'AUC', 'NLL']) + r'\\'
        print(line)
        latex.write(line + '\n')


# Curves
acc_curves = defaultdict(list)
nll_curves = defaultdict(list)
for filename in experiments:
    df = pd.read_csv(filename.replace('results.json', 'rlog.csv'))
    nb_epochs = len(df)
    for category, ax_id in categories.items():
        if category in labels[filename] + ' ':
            curve, = acc[ax_id].plot(range(1, nb_epochs + 1), df['accuracy'], label=labels[filename] + ' (acc)')
            acc_curves[ax_id].append(curve)
            curve, = nll[ax_id].plot(range(1, nb_epochs + 1), df['ll_mcmc_all'], label=labels[filename] + ' (nll)')
            nll_curves[ax_id].append(curve)
for ax_id in range(len(categories)):
    acc[ax_id].legend(handles=acc_curves[ax_id])
    nll[ax_id].legend(handles=nll_curves[ax_id])
plt.savefig('{:s}/{:s}-results.pdf'.format(CSV_FOLDER, DATASET_NAME), format='pdf')
plt.show()
