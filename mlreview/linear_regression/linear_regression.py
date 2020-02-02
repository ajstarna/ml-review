from ..utils.evaluation import mean_squared_error
from ..utils.normalization import normalize
import numpy as np

class LinearRegressor:

    def __init__(self, max_iters=5000, learning_rate=0.001, stopping_delta=0.0001):
        '''
        max_iters the maximum number of updates we will make to our weights and bias
            - I found that 1000 iters wasn't enough for the sklearn data sets, but 5k decent
        learning_rate is how fast we adjust our parameters each step
        stopping_delta is how similar the cost of two consecutive predictions needs to be before we stop early
        '''
        self.max_iters = max_iters
        self.learning_rate = learning_rate
        self.stopping_delta = stopping_delta

    def fit(self, X, Y):
        assert(len(X) == len(Y))
        num_features = X.shape[1]
        self.thetas = np.zeros((num_features, 1)) # np.random.rand(num_features, 1)
        self.mean, self.std, X = normalize(X)
        self.b = 0 #np.random.rand()
        print("X")
        print(X)
        print(X.shape)
        print(X.dtype)
        print("Y")
        print(Y)
        print(Y.shape)
        Y.shape = (Y.shape[0], 1)
        predictions = X.dot(self.thetas) + self.b
        print('predictions')
        print(predictions)
        cost = mean_squared_error(predictions, Y)
        print(cost)

        for iter in range(self.max_iters):
            #print(f'iter = {iter}')
            #print(f'b = {self.b}')
            #print(f'thetas = {self.thetas}')
        
            diffs = Y - predictions
            grad_b = (-1/X.shape[0]) * np.sum(diffs)
            #print(f'grad_b = {grad_b}')
            self.b -= grad_b * self.learning_rate

            # sum each column, since each column represents each m example for a single theta_i
            grad_theta = (-1/X.shape[0]) * np.sum( X * diffs, axis=0)
            grad_theta.shape = (num_features,1)
            self.thetas -= grad_theta * self.learning_rate
            predictions = X.dot(self.thetas) + self.b
            new_cost = mean_squared_error(predictions, Y)
            if abs(new_cost - cost) < self.stopping_delta:
                print('stopping early since the new cost is too similar to the old cost')
                break
            cost = new_cost


    def predict(self, X):
        X = (X - self.mean) / self.std  #need to apply same normalization as during training
        predictions = X.dot(self.thetas) + self.b
        print('predictions')
        print(predictions)
        return predictions

