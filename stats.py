from scipy.sparse import load_npz
from collections import defaultdict
from config import ARTICLE_FOLDER
import pandas as pd
import numpy as np
import yaml
import glob
import os

data = defaultdict(list)
values = {}
for filename in glob.glob('data/*/*'):
    if 'mangaki' in filename or '0/' not in filename:
        continue
    dataset_name = filename.split('/')[1][:-1]
    if filename.endswith('qmatrix.npz'):
        q = load_npz(filename)
        _, values[dataset_name, 'nb_skills'] = q.shape
        values[dataset_name, 'nb_skills_per_item'] = q.sum(axis=1).mean()
        print(filename, q.shape, q.sum(axis=1).mean())
    elif filename.endswith('.csv'):
        df = pd.read_csv(filename, names=['user', 'item', 'outcome', 'wins', 'fails'])
        if not 'val' in filename:
            data[dataset_name].append(df)
            print(filename, df.shape)
    elif filename.endswith('.yml'):
        with open(filename) as f:
            config = yaml.load(f)
            print(filename)

datasets = []
for dataset_name in data:
    df = pd.concat(data[dataset_name])
    nb_users, nb_items, _, _, _ = 1 + df.max()
    nb = df.groupby(['user', 'item']).count()['outcome']
    nb_entries = len(nb)
    nb_attempts_per_user = nb.mean()
    datasets.append([
        dataset_name,
        nb_users,
        nb_items,
        values.get((dataset_name, 'nb_skills'), nb_items),
        values.get((dataset_name, 'nb_skills_per_item'), 1),
        len(df),
        nb_entries / (nb_users * nb_items),
        nb_attempts_per_user
    ])
ds = pd.DataFrame(datasets, columns=('Name', 'Users', 'Items', 'Skills', 'Skills per item', 'Entries', 'Sparsity (user, item)', 'Attempts per user')).round(3).sort_values('Entries')
ds['Users'] = ds['Users'].astype(np.int32)
ds['Items'] = ds['Items'].astype(np.int32)
ds.to_csv('/tmp/datasets.csv', index=False)
ds.to_latex('/tmp/datasets.tex', index=False)
os.system('cat /tmp/datasets.csv')
os.system('cp /tmp/datasets.tex {:s}'.format(ARTICLE_FOLDER))
