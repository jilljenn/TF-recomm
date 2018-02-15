from config import *
from scipy.sparse import lil_matrix, save_npz, load_npz
import pandas as pd
import dataio
import pickle
import numpy as np

os.environ['LIBFM_PATH'] = '/Users/jin/code/libfm/bin/'

df_train, df_val, df_test = dataio.get_data()
X_train = load_npz('X_train.npz')
X_test = load_npz('X_test.npz')

with open('fm.pickle', 'rb') as f:
    bundle = pickle.load(f)
    V = bundle['V']
    V2 = np.power(V, 2)
    W = bundle['W']
    mu = bundle['mu']

def fma(x):
    return mu + x.dot(W) + 0.5 * (np.linalg.norm(x.dot(V), axis=1) ** 2 - x.dot(V2).sum(axis=1).A1)

print(X_train[:2])
print(fma(X_train[:2]))
print(X_train[:5])
print(fma(X_train[:5]))

from als3 import MangakiALS3

# als = MangakiALS3(nb_iterations=500)
# als.nb_users = USER_NUM
# als.nb_works = ITEM_NUM
# print(USER_NUM, ITEM_NUM)
# X_tr = np.array(df_train[['user', 'item']])
# y_tr = df_train['outcome']
# X_te = np.array(df_test[['user', 'item']])
# y_te = df_test['outcome']
# print(list(map(lambda x: x.shape, (X_tr, y_tr, X_te, y_te))))
# print(X_tr.max(axis=0), X_te.max(axis=0))
# print(X_tr.min(axis=0), X_te.min(axis=0))
# print(len(df_train['user'].unique()), len(df_train['item'].unique()))
# als.fit(X_tr, y_tr, y_te, X_te)
# als.save('als.pickle')
# als.compute_all_errors(X_tr, y_tr, X_te, y_te)

def get_ranking(encoded_user_id):
    X = lil_matrix((ITEM_NUM, USER_NUM + ITEM_NUM))
    for i in range(ITEM_NUM):
        # X[i, 1] = 1
        X[i, encoded_user_id] = 1
        X[i, USER_NUM + i] = 1
    print(X[:5])
    print(fma(X[:5]), 'humpf')
    rates = fma(X)
    print(rates.shape)
    print(rates[:5])
    print(rates[-5:])
    titles = pd.read_csv('/tmp/works.csv', names=('encoded_work_id', 'title'))
    titles['rates'] = rates
    return titles[['title', 'rates']].sort_values('rates', ascending=False).head(50)

# print(get_ranking(1)[-50:])

print(get_ranking(1)[-50:])
