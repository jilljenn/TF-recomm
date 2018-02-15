import yaml
import os.path


with open('/tmp/config.yml') as f:
    config = yaml.load(f)

BATCH_SIZE = 500
BATCH_SIZE = config['BATCH_SIZE']
# Keskar et al. 2016 ? Generalization gap
USER_NUM = config['USER_NUM']
ITEM_NUM = config['ITEM_NUM']
NB_CLASSES = config['NB_CLASSES']

DIM = 4
EPOCH_MAX = 10000
# LEARNING_RATE = 5 * 1e-3  # If ordinal
# LEARNING_RATE = 1e-3
# LEARNING_RATE = 5 * 1e-3 # If Adam
LEARNING_RATE = 0.1 # If Adam
LAMBDA_REG = 0.001
DISCRETE = True

DEVICE = "/cpu:0"
PREFIX = '' # + 'normalized_'
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
