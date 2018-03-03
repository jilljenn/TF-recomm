from config import LIBFM_PATH
import argparse
from scipy.sparse import lil_matrix, coo_matrix, save_npz, load_npz, hstack, diags
from sklearn.metrics import roc_auc_score, accuracy_score, log_loss
from sklearn.model_selection import KFold
import pandas as pd
import numpy as np
import os.path
import dataio
import pywFM
import json


os.environ['LIBFM_PATH'] = LIBFM_PATH

parser = argparse.ArgumentParser(description='Run Knowledge Tracing Machines')
parser.add_argument('--dataset', type=str, nargs='?', default='dummy')
parser.add_argument('--d', type=int, nargs='?')
parser.add_argument('--users', type=bool, nargs='?', const=True, default=False)
parser.add_argument('--items', type=bool, nargs='?', const=True, default=False)
parser.add_argument('--skills', type=bool, nargs='?', const=True, default=False)
parser.add_argument('--attempts', type=bool, nargs='?', const=True, default=False)
parser.add_argument('--wins', type=bool, nargs='?', const=True, default=False)
parser.add_argument('--fails', type=bool, nargs='?', const=True, default=False)
parser.add_argument('--item_wins', type=bool, nargs='?', const=True, default=False)
parser.add_argument('--item_fails', type=bool, nargs='?', const=True, default=False)
parser.add_argument('--iter', type=int, nargs='?', default=1000)
options = parser.parse_args()
DATASET_NAME = options.dataset
CSV_FOLDER, CSV_ALL, CONFIG_FILE, Q_NPZ, SKILL_WINS, SKILL_FAILS = dataio.build_new_paths(DATASET_NAME)

config = dataio.get_config(CONFIG_FILE)
USER_NUM = config['USER_NUM']
ITEM_NUM = config['ITEM_NUM']
experiment_args = vars(options)

df = dataio.get_new_data(DATASET_NAME)
try:
    qmatrix = load_npz(Q_NPZ)
except FileNotFoundError:
    qmatrix = diags([1] * ITEM_NUM).tocsr()
try:
    skill_wins = load_npz(SKILL_WINS)
    skill_fails = load_npz(SKILL_FAILS)
except:
    skill_wins = None
    skill_fails = None

short_legend, full_legend, latex_legend, active_agents = dataio.get_legend(experiment_args)
EXPERIMENT_FOLDER = os.path.join(CSV_FOLDER, short_legend)
dataio.prepare_folder(EXPERIMENT_FOLDER)
for run_id in range(5):
    dataio.prepare_folder(os.path.join(EXPERIMENT_FOLDER, str(run_id)))


def df_to_sparse(df, filename):
    SPARSE_NPZ = os.path.join(EXPERIMENT_FOLDER, filename)
    # if os.path.isfile(SPARSE_NPZ):
    #     X_fm = load_npz(SPARSE_NPZ)
    #     return X_fm
    X = {}
    nb_events, _ = df.shape
    rows = list(range(nb_events))
    X['users'] = coo_matrix(([1] * nb_events, (rows, df['user'])), shape=(nb_events, USER_NUM))
    X['items'] = coo_matrix(([1] * nb_events, (rows, df['item'])), shape=(nb_events, ITEM_NUM))
    X['skills'] = qmatrix[df['item']]

    item_wins = diags(df['wins'])
    item_fails = diags(df['fails'])
    X['item_wins'] = item_wins @ X['items']
    X['item_fails'] = item_fails @ X['items']

    if skill_wins is not None:
        print('skill wins', skill_wins.shape)
        print('skill fails', skill_fails.shape)
        print('skill attempts', (skill_wins + skill_fails).shape)
        X['attempts'] = skill_wins + skill_fails
        X['wins'] = skill_wins
        X['fails'] = skill_fails
    print([(agent, X[agent].shape) for agent in {'users', 'items', 'skills', 'attempts', 'wins', 'fails'} if agent in X])
    X_fm = hstack([X[agent] for agent in active_agents]).tocsr()
    save_npz(SPARSE_NPZ, X_fm)
    return X_fm


X_fm = df_to_sparse(df, 'X.npz')

params = {
    'task': 'classification',
    'num_iter': options.iter,
    'rlog': True,
    'learning_method': 'mcmc'
}
if options.d > 0:
    params['k2'] = options.d

# Run experiments
kf = KFold(n_splits=5, shuffle=True)
for run_id, (i_train, i_test) in enumerate(kf.split(X_fm)):
    X_train = X_fm[i_train]
    y_train = df.iloc[i_train]['outcome']
    X_test = X_fm[i_test]
    y_test = df.iloc[i_test]['outcome']

    fm = pywFM.FM(**params)
    model = fm.run(X_train, y_train, X_test, y_test)

    ACC = accuracy_score(y_test, np.round(model.predictions))
    print('accuracy', ACC)
    AUC = roc_auc_score(y_test, model.predictions)
    NLL = log_loss(y_test, model.predictions)

    model.rlog.to_csv(os.path.join(EXPERIMENT_FOLDER, str(run_id), 'rlog.csv'))
    with open(os.path.join(EXPERIMENT_FOLDER, str(run_id), 'results.json'), 'w') as f:
        f.write(json.dumps({
            'args': experiment_args,
            'legends': {
                'short': short_legend,
                'full': full_legend,
                'latex': latex_legend
            },
            'metrics': {
                'ACC': ACC,
                'AUC': AUC,
                'NLL': NLL
            }
        }, indent=4))
