import yaml
import os.path


DATASET_NAME = 'assistments'
# DATASET_NAME = 'berkeley'
# DATASET_NAME = 'castor6e'
# DATASET_NAME = 'fraction'
# DATASET_NAME = 'timss2003'


DATA_FOLDER = 'data'
CSV_FOLDER = os.path.join(DATA_FOLDER, DATASET_NAME)
CSV_TRAIN = os.path.join(CSV_FOLDER, 'train.csv')
CSV_TEST = os.path.join(CSV_FOLDER, 'test.csv')
CSV_VAL = os.path.join(CSV_FOLDER, 'val.csv')
CONFIG = os.path.join(CSV_FOLDER, 'config.yml')
Q_NPZ = os.path.join(CSV_FOLDER, 'qmatrix.npz')


with open(CONFIG) as f:
    config = yaml.load(f)

# BATCH_SIZE = 500  # Keskar et al. 2016 ? Generalization gap
BATCH_SIZE = config['BATCH_SIZE']
USER_NUM = config['USER_NUM']
ITEM_NUM = config['ITEM_NUM']
NB_CLASSES = config['NB_CLASSES']

DIM = 20
EPOCH_MAX = 10000
# LEARNING_RATE = 5 * 1e-3  # If ordinal
# LEARNING_RATE = 1e-3
LEARNING_RATE = 5 * 1e-3
# LEARNING_RATE = 0.01 # If Adam
LAMBDA_REG = 0.01
DISCRETE = True

DEVICE = "/cpu:0"
PREFIX = '' # + 'normalized_'
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

ARTICLE_FOLDER = '/Users/jilljenn/code/article/edm2018/'
LIBFM_PATH = '/Users/jilljenn/code/libfm/bin/'
