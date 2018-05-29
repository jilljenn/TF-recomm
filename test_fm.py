import pywFM
import numpy as np
import pandas as pd
import os

os.environ['LIBFM_PATH'] = '/Users/jilljenn/code/libfm/bin/'

features = np.matrix([
#     Users  |     Movies     |    Movie Ratings   | Time | Last Movies Rated
#    A  B  C | TI  NH  SW  ST | TI   NH   SW   ST  |      | TI  NH  SW  ST
    [1, 0, 0,  1,  0,  0,  0,   0.3, 0.3, 0.3, 0,     13,   0,  0,  0,  0 ],
    [1, 0, 0,  0,  1,  0,  0,   0.3, 0.3, 0.3, 0,     14,   1,  0,  0,  0 ],
    [1, 0, 0,  0,  0,  1,  0,   0.3, 0.3, 0.3, 0,     16,   0,  1,  0,  0 ],
    [0, 1, 0,  0,  0,  1,  0,   0,   0,   0.5, 0.5,   5,    0,  0,  0,  0 ],
    [0, 1, 0,  0,  0,  0,  1,   0,   0,   0.5, 0.5,   8,    0,  0,  1,  0 ],
    [0, 0, 1,  1,  0,  0,  0,   0.5, 0,   0.5, 0,     9,    0,  0,  0,  0 ],
    [0, 0, 1,  0,  0,  1,  0,   0.5, 0,   0.5, 0,     12,   1,  0,  0,  0 ]
])
target = [0, 1, 1, 0, 0, 0, 1]

fm = pywFM.FM(task='classification', num_iter=50, rlog=False)

# split features and target for train/test
# first 5 are train, last 2 are test
model = fm.run(features[:5], target[:5], features[5:], target[5:])
print(model.predictions)
# you can also get the model weights
print(model.weights)
