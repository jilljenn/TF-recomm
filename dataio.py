from __future__ import absolute_import, division, print_function
import os
import numpy as np
import pandas as pd
import yaml


def build_paths(DATASET_NAME):
    DATA_FOLDER = 'data'
    CSV_FOLDER = os.path.join(DATA_FOLDER, DATASET_NAME)
    CONFIG_FILE = os.path.join(CSV_FOLDER, 'config.yml')
    CSV_TRAIN = os.path.join(CSV_FOLDER, 'train.csv')
    CSV_TEST = os.path.join(CSV_FOLDER, 'test.csv')
    CSV_VAL = os.path.join(CSV_FOLDER, 'val.csv')
    Q_NPZ = os.path.join(CSV_FOLDER, 'qmatrix.npz')
    return CSV_FOLDER, CSV_TRAIN, CSV_TEST, CSV_VAL, CONFIG_FILE, Q_NPZ


def build_new_paths(DATASET_NAME):
    DATA_FOLDER = 'data'
    CSV_FOLDER = os.path.join(DATA_FOLDER, DATASET_NAME)
    CONFIG_FILE = os.path.join(CSV_FOLDER, 'config.yml')
    CSV_ALL = os.path.join(CSV_FOLDER, 'all.csv')
    Q_NPZ = os.path.join(CSV_FOLDER, 'qmatrix.npz')
    SKILL_WINS = os.path.join(CSV_FOLDER, 'skill_wins.npz')
    SKILL_FAILS = os.path.join(CSV_FOLDER, 'skill_fails.npz')
    return CSV_FOLDER, CSV_ALL, CONFIG_FILE, Q_NPZ, SKILL_WINS, SKILL_FAILS


def get_config(CONFIG_FILE):
    with open(CONFIG_FILE) as f:
        config = yaml.load(f)
    # BATCH_SIZE = 500  # Keskar et al. 2016 ? Generalization gap
    return config


def read_process(filename, sep="\t"):
    # col_names = ["user", "item", "outcome"]
    col_names = ["user", "item", "outcome", "wins", "fails"]
    # col_names = ["users", "items", "speech", "outcome", "wins", "fails"]
    df = pd.read_csv(filename, sep=sep, header=None, names=col_names, engine='python')
    for col in {"user", "item"}:
        df[col] = df[col].astype(np.int32)
    df["outcome"] = df["outcome"].astype(np.float32)
    return df


def get_data(DATASET_NAME):
    CSV_FOLDER, CSV_TRAIN, CSV_TEST, CSV_VAL, CONFIG, Q_NPZ = build_paths(DATASET_NAME)
    df_train = read_process(CSV_TRAIN, sep=",")
    df_val = read_process(CSV_VAL, sep=",")
    df_test = read_process(CSV_TEST, sep=",")
    return df_train, df_val, df_test


def get_new_data(DATASET_NAME):
    CSV_FOLDER, CSV_ALL, CONFIG_FILE, Q_NPZ, SKILL_WINS, SKILL_FAILS = build_new_paths(DATASET_NAME)
    df = read_process(CSV_ALL, sep=",")
    return df


def get_legend(experiment_args):
    dim = experiment_args['d']
    short = ''
    full = ''
    agents = ['users', 'items', 'skills', 'attempts', 'wins', 'fails', 'item_wins', 'item_fails']
    active = []
    for agent in agents:
        if experiment_args.get(agent):
            short += agent[0] if '_' not in agent else ('W' if '_w' in agent else 'F')
            active.append(agent)
    short += str(dim)
    prefix = ''
    if set(active) == {'users', 'items'} and dim == 0:
        prefix = 'IRT: '
    elif set(active) == {'users', 'items'} and dim > 0:
        prefix = 'MIRTb: '
    elif set(active) == {'users', 'skills', 'attempts'} and dim == 0:
        prefix = 'AFM: '
    elif set(active) == {'users', 'skills', 'wins', 'fails'} and dim == 0:
        prefix = 'PFA: '
    full = prefix + ', '.join(active) + ' d = {:d}'.format(dim)
    latex = prefix + ', '.join(active)#.replace('_', r'\_')
    return short, full, latex, active


def prepare_folder(path):
    if not os.path.isdir(path):
        os.makedirs(path)


class ShuffleIterator(object):
    """
    Randomly generate batches
    """
    def __init__(self, inputs, batch_size=10):
        self.inputs = inputs
        self.batch_size = batch_size
        self.num_cols = len(self.inputs)
        self.len = len(self.inputs[0])
        self.inputs = np.transpose(np.vstack([np.array(self.inputs[i]) for i in range(self.num_cols)]))

    def __len__(self):
        return self.len

    def __iter__(self):
        return self

    def __next__(self):
        return self.next()

    def next(self):
        ids = np.random.randint(0, self.len, (self.batch_size,))
        out = self.inputs[ids, :]
        return [out[:, i] for i in range(self.num_cols)]


class OneEpochIterator(ShuffleIterator):
    """
    Sequentially generate one-epoch batches, typically for test data
    """
    def __init__(self, inputs, batch_size=10):
        super(OneEpochIterator, self).__init__(inputs, batch_size=batch_size)
        if batch_size > 0:
            self.idx_group = np.array_split(np.arange(self.len), np.ceil(self.len / batch_size))
        else:
            self.idx_group = [np.arange(self.len)]
        self.group_id = 0

    def next(self):
        if self.group_id >= len(self.idx_group):
            self.group_id = 0
            raise StopIteration
        out = self.inputs[self.idx_group[self.group_id], :]
        self.group_id += 1
        return [out[:, i] for i in range(self.num_cols)]
