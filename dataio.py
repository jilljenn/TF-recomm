from __future__ import absolute_import, division, print_function
from config import CSV_TRAIN, CSV_VAL, CSV_TEST
import os
import numpy as np
import pandas as pd


def read_process(filname, sep="\t"):
    # col_names = ["user", "item", "outcome"]
    col_names = ["user", "item", "outcome", "wins", "fails"]
    df = pd.read_csv(filname, sep=sep, header=None, names=col_names, engine='python')
    # df["user"] -= 1
    # df["item"] -= 1
    for col in {"user", "item"}:
        df[col] = df[col].astype(np.int32)
    df["outcome"] = df["outcome"].astype(np.float32)
    return df


def get_data():
    df_train = read_process(CSV_TRAIN, sep=",")
    df_val = read_process(CSV_VAL, sep=",")
    df_test = read_process(CSV_TEST, sep=",")
    return df_train, df_val, df_test


def get_legend(options):
    short = ''
    full = ''
    agents = ['users', 'items', 'skills']
    active = []
    for agent in agents:
        if vars(options)[agent]:
            short += agent[0]
            active.append(agent)
    short += str(options.d)
    full = ', '.join(active) + ' d = {:d}'.format(options.d)
    latex = r'\textnormal{{:s} } d = {:d}'.format(', '.join(active), options.d)
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
