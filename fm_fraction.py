from config import *
import argparse
from scipy.sparse import lil_matrix, coo_matrix, save_npz, load_npz, hstack
from sklearn.metrics import roc_auc_score, accuracy_score, log_loss
import pandas as pd
import numpy as np
import dataio
import pywFM
import json


os.environ['LIBFM_PATH'] = '/Users/jilljenn/code/libfm/bin/'

df_train, df_val, df_test = dataio.get_data()
qmatrix = load_npz(Q_NPZ)
_, SKILL_NUM = qmatrix.shape

parser = argparse.ArgumentParser(description='Run FM on non-temporal data')
parser.add_argument('--d', type=int, nargs='?')
parser.add_argument('--users', type=bool, nargs='?', const=True, default=False)
parser.add_argument('--items', type=bool, nargs='?', const=True, default=False)
parser.add_argument('--skills', type=bool, nargs='?', const=True, default=False)
options = parser.parse_args()


short_legend, full_legend, latex_legend, active_agents = dataio.get_legend(options)
EXPERIMENT_FOLDER = os.path.join(CSV_FOLDER, short_legend)
dataio.prepare_folder(EXPERIMENT_FOLDER)


def df_to_sparse(df, filename):
    SPARSE_NPZ = os.path.join(EXPERIMENT_FOLDER, filename)
    if os.path.isfile(SPARSE_NPZ):
        X_fm = load_npz(SPARSE_NPZ)
        return X_fm
    X = {}
    nb_events, _ = df.shape
    rows = list(range(nb_events))
    X['users'] = coo_matrix(([1] * nb_events, (rows, df['user'])), shape=(nb_events, USER_NUM))
    X['items'] = coo_matrix(([1] * nb_events, (rows, df['item'])), shape=(nb_events, ITEM_NUM))
    X['skills'] = qmatrix[df['item']]
    print([(agent, X[agent].shape) for agent in {'users', 'items', 'skills'}])
    X_fm = hstack([X[agent] for agent in active_agents]).tocsr()
    save_npz(SPARSE_NPZ, X_fm)
    return X_fm

X_train = df_to_sparse(df_train, 'X_train.npz')
print('Train done', X_train.shape)
X_test = df_to_sparse(df_test, 'X_test.npz')
print('Test done', X_test.shape)

params = {
    'task': 'classification',
    'num_iter': 500,
    'rlog': True,
    'learning_method': 'mcmc'
}
if options.d > 0:
    params['k2'] = options.d
fm = pywFM.FM(**params)
model = fm.run(X_train, df_train['outcome'], X_test, df_test['outcome'])

ACC = accuracy_score(df_test['outcome'], np.round(model.predictions))
AUC = roc_auc_score(df_test['outcome'], model.predictions)
NLL = log_loss(df_test['outcome'], model.predictions)

model.rlog.to_csv(os.path.join(EXPERIMENT_FOLDER, 'rlog.csv'))
with open(os.path.join(EXPERIMENT_FOLDER, 'results.json'), 'w') as f:
    f.write(json.dumps({
        'args': vars(options),
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
    }))
