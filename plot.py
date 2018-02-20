from config import *
# import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob
import json


experiments = glob.glob('%s/*/results.json' % CSV_FOLDER)
fig, axes = plt.subplots(2, 1, figsize=(8, 16), sharex='col')  # x-axis will be shared across columns
acc, nll = axes

# Tables
labels = {}
with open(os.path.join(CSV_FOLDER, 'table.tex'), 'w') as latex:
    for filename in experiments:
        with open(filename) as f:
            results = json.load(f)
        labels[filename] = results['legends']['full']
        line = '{:s} & '.format(labels[filename])
        line += ' & '.join('{:.4f}'.format(results['metrics'][metric]) for metric in ['ACC', 'AUC', 'NLL']) + r'\\' + '\n'
        print(line)
        latex.write(line)


# Curves
acc_curves = []
nll_curves = []
for filename in experiments:
    df = pd.read_csv(filename.replace('results.json', 'rlog.csv'))
    nb_epochs = len(df)
    curve, = acc.plot(range(1, nb_epochs + 1), df['accuracy'], label=labels[filename] + ' (acc)')
    acc_curves.append(curve)
    curve, = nll.plot(range(1, nb_epochs + 1), df['ll_mcmc_all'], label=labels[filename] + ' (nll)')
    nll_curves.append(curve)
acc.legend(handles=acc_curves)
nll.legend(handles=nll_curves)
plt.savefig('%s/results.pdf' % CSV_FOLDER, format='pdf')
plt.show()
