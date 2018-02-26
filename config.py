import os


# CSV_FOLDER, CSV_TRAIN, CSV_TEST, CSV_VAL, CONFIG, Q_NPZ = dataio.build_paths(DATASET_NAME)

# with open(CONFIG) as f:
#     config = yaml.load(f)

# # BATCH_SIZE = 500  # Keskar et al. 2016 ? Generalization gap
# BATCH_SIZE = config['BATCH_SIZE']
# USER_NUM = config['USER_NUM']
# ITEM_NUM = config['ITEM_NUM']
# NB_CLASSES = config['NB_CLASSES']

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
