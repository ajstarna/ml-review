'''
File for functions related to bootstrapping algorithms.
'''

import numpy as np

def get_bootstrapped_data_set(X, Y, size_of_output=None):
    if size_of_output is None:
        size_of_output = X.shape[0]

    indices_used = np.random.choice(X.shape[0], size_of_output)
    bootstrapped_data = {
        'data': X[indices_used],
        'target': Y[indices_used],
        'indices_used': indices_used
        }
    return bootstrapped_data
