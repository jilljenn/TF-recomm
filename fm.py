import argparse
from scipy.sparse import lil_matrix, coo_matrix, save_npz, load_npz, hstack, diags
from sklearn.metrics import roc_auc_score, accuracy_score, log_loss
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
import pandas as pd
import numpy as np
import os.path
import dataio
import time
import pywFM
import json
import sys


parser = argparse.ArgumentParser(description='Run Knowledge Tracing Machines')
parser.add_argument('--base_dir', type=str, nargs='?', default='/home/jj')  # /Users/jilljenn
parser.add_argument('--libfm', type=str, nargs='?', default='libfm')  # code/libfm
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
parser.add_argument('--iter', type=int, nargs='?', default=500)
options = parser.parse_args()
os.environ['LIBFM_PATH'] = os.path.join(options.base_dir, options.libfm, 'bin')
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
    if os.path.isfile(SPARSE_NPZ):  # FIXME: comment to avoid obsolete encodings
        X_fm = load_npz(SPARSE_NPZ)
        return X_fm
    X = {}
    nb_events, _ = df.shape
    rows = list(range(nb_events))
    X['users'] = coo_matrix(([1] * nb_events, (rows, df['user'])), shape=(nb_events, USER_NUM))
    X['items'] = coo_matrix(([1] * nb_events, (rows, df['item'])), shape=(nb_events, ITEM_NUM))
    X['skills'] = qmatrix[df['item']]

    X['item_wins'] = X['items'].copy()
    X['item_wins'].data = df['wins']
    X['item_fails'] = X['items'].copy()
    X['item_fails'].data = df['fails']

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


print(df.head())
X_fm = df_to_sparse(df, 'X.npz')
print('Encoding done')

params = {
    'task': 'classification',
    'num_iter': options.iter,
    'rlog': True,
    'learning_method': 'mcmc',
    'k2': options.d
}


# Run experiments by separating students
kf = KFold(n_splits=5, shuffle=True)
# for run_id, (i_train, i_test) in enumerate(kf.split(X_fm)):
all_users = df['user'].unique()
for run_id, (i_user_train, i_user_test) in enumerate(kf.split(all_users)):
    users_train = all_users[i_user_train]
    users_test = all_users[i_user_test]

    df_train = df.query('user in @users_train')
    df_test = df.query('user in @users_test')

    X_train = X_fm[list(df_train.index)]  # Without "list", takes 24 GB RAM
    y_train = df_train['outcome']
    X_test = X_fm[list(df_test.index)]
    y_test = df_test['outcome']

    start = time.time()
    if options.d == 0:
        print('fitting...')
        model = LogisticRegression()
        model.fit(X_train, y_train)
        print('fit', time.time() - start)
        y_pred_test = model.predict_proba(X_test)[:, 1]
    else:
        fm = pywFM.FM(**params)
        model = fm.run(X_train, y_train, X_test, y_test)
        print('fit', time.time() - start)
        y_pred_test = model.predictions
        model.rlog.to_csv(os.path.join(EXPERIMENT_FOLDER, str(run_id), 'rlog.csv'))

    ACC = accuracy_score(y_test, np.round(y_pred_test))
    print('acc', ACC)
    AUC = roc_auc_score(y_test, y_pred_test)
    print('auc', AUC)
    NLL = log_loss(y_test, y_pred_test)

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
