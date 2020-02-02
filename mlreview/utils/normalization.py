import numpy as np

def normalize(X):
    # normalize the data to have mean of 0 and standard deviation of 1 along each column
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    std = np.where(std==0, 1, std)  # dont want to divide by zero
    normalized = (X - mean)/ std
    return mean, std, normalized
