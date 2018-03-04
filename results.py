from collections import defaultdict
import pandas as pd
import glob
import json


array = defaultdict(dict)
experiments = glob.glob('*/*/*/results.json')
for experiment in experiments:
    with open(experiment) as f:
        results = json.load(f)
    short_legend = results['legends']['short']
    full_legend = results['legends']['full']
    shortname = full_legend.split(':')[0] if ':' in full_legend else short_legend
    array[results['args']['dataset']][shortname] = results['metrics']['AUC']
print(pd.DataFrame.from_dict(array))
