from collections import Counter, defaultdict
import math
import numpy as np
import re


def avgstd(l): # Displays mean and variance
    n = len(l)
    # print 'computed over %d values' % n
    mean = float(sum(l)) / n
    var = float(sum(i * i for i in l)) / n - mean * mean
    return '%.3f ± %.3f' % (round(mean, 3), round(1.96 * math.sqrt(var / (10 * n)), 3))
    # return mean, round(1.96 * math.sqrt(var / n), 5)


def get_scores():
    values = defaultdict(lambda: [])
    logs = {}
    with open('3-pdf-fisher') as f:
        logs['fisher'] = f.read().splitlines()
    with open('3-pdf-random') as f:
        logs['random'] = f.read().splitlines()
    with open('3-pdf-popular') as f:
        logs['popular'] = f.read().splitlines()

    r = re.compile('^([0-9]+) +([0-9]+) +([0-9]+) .*mobo=([0-9.]+), rmse=([0-9.]+),.*mobo=([0-9.]+), rmse=([0-9.]+),')
    for model in logs.keys():
        for line in logs[model]:
            m = r.match(line)
            if m:
                user_id, _, t, train_mobo, train_rmse, test_mobo, test_rmse = m.groups()
                t = int(t)
                values[model, 'train_mobo', t].append(float(train_mobo))
                values[model, 'train_rmse', t].append(float(train_rmse))
                values[model, 'test_mobo', t].append(float(test_mobo))
                values[model, 'test_rmse', t].append(float(test_rmse))

    BUDGET = 10
    for t in range(BUDGET):
        for quantity in ['test_mobo', 'test_rmse']:  # 'train_mobo', 'train_rmse', 
            for model in logs.keys():
                print(t, quantity, model, avgstd(values[model, quantity, t]))
    return values
