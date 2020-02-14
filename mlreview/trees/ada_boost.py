from .decision_tree import DecisionTreeClassifier, DecisionTreeRegressor
from ..utils.bootstrapping import get_bootstrapped_data_set

from collections import defaultdict
import numpy as np

class AdaBoostClassifier:

    def __init__(self, num_estimators=10):
        self.num_estimators = num_estimators


    def init_weights(self, X):
        # each training example gets an associated weight,
        # which starts at 1/N
        weights = np.empty(X.shape[0])
        starting_weight = 1/len(weights)
        weights.fill(starting_weight)
        return weights

    def fit(X, Y):
        weights = self.init_weights(X)
        print(weights)
        for i in range(self.num_estimators):
            print('hi')

    def predict(X):
        # all_predictions  = [tree.predict(X) for tree in self.trees]
        final = np.zeros(X.shape[0])
        '''
        for preds in all_predictions:
            final += preds
        final /= self.num_trees
        final = np.around(final)
        '''
        return final
        
        
