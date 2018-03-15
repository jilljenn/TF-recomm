from config import LIBFM_PATH
from scipy.sparse import lil_matrix, coo_matrix, save_npz, load_npz, hstack, diags
from sklearn.metrics import roc_auc_score, accuracy_score, log_loss
import numpy as np
import argparse
import os.path
import dataio
import pywFM


os.environ['LIBFM_PATH'] = LIBFM_PATH
parser = argparse.ArgumentParser(description='Run Knowledge Tracing Machines')
parser.add_argument('--dataset', type=str, nargs='?', default='dummy')
parser.add_argument('--d', type=int, nargs='?')
parser.add_argument('--users', type=bool, nargs='?', const=True, default=False)
parser.add_argument('--items', type=bool, nargs='?', const=True, default=False)
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

df_train, df_val, df_test = dataio.get_data(DATASET_NAME)
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

X_train = df_to_sparse(df_train, 'X_train.npz')
y_train = df_train['outcome']
print('Encoding train done')
X_test = df_to_sparse(df_test, 'X_test.npz')
y_test = df_test['outcome']
print('Encoding test done')

params = {
    'task': 'classification',
    'num_iter': options.iter,
    'rlog': True,
    'learning_method': 'mcmc'
}
if options.d > 0:
    params['k2'] = options.d

fm = pywFM.FM(**params)
model = fm.run(X_train, y_train, X_test, y_test)

ACC = accuracy_score(y_test, np.round(model.predictions))
AUC = roc_auc_score(y_test, model.predictions)
NLL = log_loss(y_test, model.predictions)
print('accuracy', ACC)
print('AUC', AUC)
print('NLL', NLL)
