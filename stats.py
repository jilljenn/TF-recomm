from scipy.sparse import load_npz
import pandas as pd
import glob
import os

for filename in glob.glob('data/*/*'):
    if filename.endswith('.npz'):
        q = load_npz(filename)
        print(filename, q.shape, q.sum(axis=1).mean())
    elif filename.endswith('.csv'):
        df = pd.read_csv(filename)
        print(filename, df.shape)
    elif filename.endswith('.yml'):
        os.system('cat {}'.format(filename))
