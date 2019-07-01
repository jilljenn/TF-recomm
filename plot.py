import argparse
from cycler import cycler
from config import ARTICLE_FOLDER
from collections import defaultdict
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# plt.margins(x=0, y=0)
params = {'text.usetex': True, 'font.family': 'serif'}
plt.rcParams.update(params)
import dataio
import glob
import json
import os.path


# interesting = [
#     'users, items, skills d = 0',
#     'users, items, attempts d = 0',
#     'users, items, skills, wins, fails, item_wins, item_fails d = 1',
#     'users, items, skills d = 10',
#     'users, skills d = 5',
#     'PFA: users, skills, wins, fails d = 0',
#     'IRT: users, items d = 0',
#     'MIRTb: users, items d = 10',
#     'MIRTb: users, items d = 20',
#     'AFM: users, skills, attempts d = 0'
# ]

cache_styles = {}
default_cycler = (plt.rcParams['axes.prop_cycle'] +
                  cycler(linestyle=['-', '--', ':', '-.', '-', '--', ':', '-.', '-', '--']) +
                  cycler(lw=[1, 1, 1, 1, 2, 2, 2, 2, 3, 3]))
def get_style(legend):
    if legend in cache_styles:
        return cache_styles[legend]
    # if 'IRT:' in legend:
    #     return {'color': 'red', 'linewidth': 1, 'linestyle': ':'}
    # elif 'MIRTb:' in legend:
    #     return {'color': 'red', 'linewidth': 2, 'linestyle': ':'}
    # elif 'PFA:' in legend:
    #     return {'color': 'blue', 'linewidth': 1, 'linestyle': '--'}
    style = list(default_cycler)[len(cache_styles) % len(default_cycler)]
    cache_styles[legend] = style
    return style

parser = argparse.ArgumentParser(description='Generate tables and plots')
parser.add_argument('--dataset', type=str, nargs='+')
options = parser.parse_args()
for DATASET_NAME in options.dataset:
    print(DATASET_NAME)
    CSV_FOLDER, CSV_TRAIN, CSV_TEST, CSV_VAL, CONFIG_FILE, Q_NPZ = dataio.build_paths(DATASET_NAME)

    if DATASET_NAME.startswith('assistments') or DATASET_NAME.startswith('berkeley'):
        categories = ['Effect of data', 'Effect of dimension']
        categories_regexp = ['d = 10', 'items, skills, wins, fails, extra']
    else:
        categories = ['Effect of data', 'Effect of dimension']
        categories_regexp = ['d = 10', 'items, skills, wins, fails']
    # elif DATASET_NAME.startswith('fraction'):
    #     categories = ['All models', 'users + items', 'd = 0']
    #     categories_regexp = ['', 'users, items ', 'd = 0 ']
    # elif DATASET_NAME.startswith('timss2003'):
    #     categories = ['users + items + skills', 'd = 5']
    #     categories_regexp = ['users, items, skills ', 'd = 5 ']
    # elif DATASET_NAME.startswith('castor6e'):
    #     categories = ['users + items + skills', 'd = 5']
    #     categories_regexp = ['users, items, skills', 'd = 5 ']


    experiments = glob.glob('%s/*/results.json' % CSV_FOLDER)
    fig, axes = plt.subplots(1, len(categories), figsize=(8, 4), sharex='col', sharey='row')  # x-axis will be shared across columns
    # acc, nll = axes
    acc = axes

    # Tables
    labels = {}
    latex_labels = {}
    TABLE_TEX = os.path.join(CSV_FOLDER, '{:s}-table.tex'.format(DATASET_NAME))
    array = []
    acc_values = {}
    nll_values = {}
    #with open(TABLE_TEX, 'w') as latex:
    for filename in experiments:
        fm_log_file = filename.replace('results.json', 'rlog.csv')
        if os.path.isfile(fm_log_file) and ('assistments' in filename or 'berkeley' in filename):
            df = pd.read_csv(fm_log_file)
            acc_values[filename] = df['accuracy']
            nll_values[filename] = df['ll_mcmc_all']
        with open(filename) as f:
            results = json.load(f)
        short_legend, full_legend, latex_legend, active_agents = dataio.get_legend(results['args'])
        # full_legend = full_legend
        latex_legend = latex_legend.replace('users, items, skills, wins, fails, item_wins, item_fails', 'u, i, s, w, f, item_wins, item_fails').replace('_', r'\_')
        print(filename)
        print(latex_legend)
        print(full_legend)
        labels[filename] = full_legend
        latex_labels[filename] = latex_legend + " $d = {:d}$".format(results['args']['d'])
        # line = '{:s} & '.format(latex_legend)
        # line += ' & '.join('{:.4f}'.format(results['metrics'][metric]) for metric in ['ACC', 'AUC', 'NLL']) + r'\\'
        row = [latex_legend, results['args']['d']] + [results['metrics'][metric] for metric in ['ACC', 'AUC', 'NLL']]
        # MCMC NLL is different
        # row[-1] = nll_values.get(filename).values[-1]
        array.append(row)
        # latex.write(line + '\n')
    pd.set_option('display.max_colwidth', -1)
    df = pd.DataFrame(array, columns=('model', 'dim', 'ACC', 'AUC', 'NLL'))
    df = df.sort_values('AUC', ascending=False).round(3)
    for metric in {'ACC', 'AUC', 'NLL'}:
        extremum = df[metric].max() if metric in {'ACC', 'AUC'} else df[metric].min()
        winners = df.query('abs({:s} - @extremum) <= 0.0011'.format(metric))
        df.loc[winners.index, metric] = winners[metric].map(lambda entry: r'\textbf{{{:.3f}}}'.format(entry))
    df.to_latex(TABLE_TEX, column_format='c' * 5, escape=False, index=False)

    # Curves
    acc_curves = defaultdict(list)
    acc_labels = defaultdict(list)
    nll_curves = defaultdict(list)
    for filename in experiments:
        if filename in acc_values:
            nb_epochs = len(acc_values[filename])
            for ax_id, category_regexp in enumerate(categories_regexp):
                if category_regexp in labels[filename] + ' ' and (category_regexp != '' or True):#labels[filename] in interesting):
                    curve, = acc[ax_id].plot(range(1, nb_epochs + 1), acc_values[filename], label=latex_labels[filename], **get_style(labels[filename]))
                    acc_curves[ax_id].append(curve)
                    acc_labels[ax_id].append(labels[filename])
                    # curve, = nll[ax_id].plot(range(1, nb_epochs + 1), nll_values[filename], label=latex_labels[filename], **get_style(labels[filename]))
                    # nll_curves[ax_id].append(curve)
    for ax_id, category in enumerate(categories):
        acc[ax_id].legend(handles=acc_curves[ax_id])
        acc[ax_id].set_title(category)
        # nll[ax_id].legend(handles=nll_curves[ax_id])
        # nll[ax_id].set_xlabel('Epochs')
    #handles, labels = axes.get_legend_handles_labels()
    #fig.legend(acc_curves[0], acc_labels[0], loc='upper center')
    acc[0].set_ylabel('Accuracy')
    acc[0].set_xlabel('Epochs')
    acc[1].set_xlabel('Epochs')
    # nll[0].set_ylabel('Log-loss')
    RESULTS_PDF = os.path.join(CSV_FOLDER, '{:s}-results.pdf'.format(DATASET_NAME))
    #plt.suptitle(DATASET_NAME[:-1].title())
    plt.savefig(RESULTS_PDF, format='pdf', bbox_inches='tight', pad_inches=0)
    #os.system('open {:s}'.format(TABLE_TEX))
    os.system('open {:s}'.format(RESULTS_PDF))
    # os.system('cp {:s} {:s}tables/'.format(TABLE_TEX, ARTICLE_FOLDER))
    # os.system('cp {:s} {:s}figures/'.format(RESULTS_PDF, ARTICLE_FOLDER))
    # plt.show()
