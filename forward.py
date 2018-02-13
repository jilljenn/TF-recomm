from config import *
from scipy.sparse import lil_matrix, save_npz, load_npz
import dataio
import pickle
import numpy as np

os.environ['LIBFM_PATH'] = '/Users/jin/code/libfm/bin/'

df_train, df_val, df_test = dataio.get_data()
X_train = load_npz('X_train.npz')
X_test = load_npz('X_test.npz')

with open('fm.pickle', 'rb') as f:
    bundle = pickle.load(f)

print(X_test[0], df_test['outcome'][0])
x = X_test[0]
print(bundle['mu'] + np.array(bundle['W']).dot(x) + np.array(bundle['V'][1396]).dot(np.array(bundle['V'][3739])))
