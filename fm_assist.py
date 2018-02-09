from config import *
from scipy.sparse import lil_matrix, save_npz, load_npz
from sklearn.metrics import roc_auc_score
import pandas as pd
import numpy as np
import dataio
import pywFM

os.environ['LIBFM_PATH'] = '/Users/jilljenn/code/libfm/bin/'

df_train, df_val, df_test = dataio.get_data()

def df_to_sparse(df, filename):
    nb_events, _ = df.shape
    X = lil_matrix((nb_events, USER_NUM + 3 * ITEM_NUM))
    for i, (user_id, item_id, outcome, nb_wins, nb_fails) in enumerate(np.array(df)):
        X[i, user_id] = 1
        X[i, USER_NUM + item_id] = 1
        X[i, USER_NUM + ITEM_NUM + item_id] = nb_wins
        X[i, USER_NUM + 2 * ITEM_NUM + item_id] = nb_fails
    save_npz(filename, X.tocsr())

# df_to_sparse(df_train, 'X_train.npz')
# print('Train done')
# df_to_sparse(df_test, 'X_test.npz')
# print('Test done')

X_train = load_npz('X_train.npz')
X_test = load_npz('X_test.npz')
print(X_train.shape)
print(X_test.shape)

fm = pywFM.FM(task='classification', num_iter=100, k2=1, rlog=False, learning_method='mcmc', r1_regularization=0.1, r2_regularization=0.1)
model = fm.run(X_train, df_train['outcome'], X_test, df_test['outcome'])
#print(model)
print(roc_auc_score(df_test['outcome'], model.predictions))
