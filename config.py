import os


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
# LIBFM_PATH = '/Users/jilljenn/code/libfm/bin/'
