import yaml
import os.path


with open('/tmp/config.yml') as f:
    config = yaml.load(f)

BATCH_SIZE = config['BATCH_SIZE'] // 2 # 100000  # config['BATCH_SIZE']
USER_NUM = config['USER_NUM']
ITEM_NUM = config['ITEM_NUM']
NB_CLASSES = config['NB_CLASSES']

DIM = 20
EPOCH_MAX = 200
#LEARNING_RATE = 5 * 1e-3  # If dim == 0
LEARNING_RATE = 0.1  # If Adam
LAMBDA_REG = 0.
DISCRETE = True

DEVICE = "/cpu:0"
PREFIX = '' # + 'normalized_'
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
