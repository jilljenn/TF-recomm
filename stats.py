from scipy.sparse import load_npz
import pandas as pd
import yaml
import glob
import os

for filename in glob.glob('data/*/*'):
    if 'mangaki' in filename:
        continue
    if filename.endswith('.npz'):
        q = load_npz(filename)
        print(filename, q.shape, q.sum(axis=1).mean())
    elif filename.endswith('.csv'):
        df = pd.read_csv(filename, names='12345')
        print(filename, df.shape)
    elif filename.endswith('.yml'):
        with open(filename) as f:
            config = yaml.load(f)
            print(filename)
            print('Entries', config['TRAIN'] + config['TEST'])
            print('Sparsity', (config['TRAIN'] + config['TEST']) / (config['USER_NUM'] * config['ITEM_NUM']))
