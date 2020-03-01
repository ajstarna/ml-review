'''
http://www.site.uottawa.ca/~stan/csi5387/boost-tut-ppr.pdf
'''

from ..trees.decision_tree import DecisionTreeStump
#from ..utils.bootstrapping import get_bootstrapped_data_set

from collections import defaultdict
import numpy as np

class AdaBoostClassifier:

    def __init__(self, num_iters=10):
        self.num_iters = num_iters

    def set_index_to_feature_type(self, index_to_feature_type):
        self.index_to_feature_type = index_to_feature_type

    def init_weights(self, X):
        # each training example gets an associated weight,
        # which starts at 1/N
        weights = np.empty(X.shape[0])
        starting_weight = 1/len(weights)
        weights.fill(starting_weight)
        return weights

    def fit(self, X, Y):
        weights = self.init_weights(X) # the weights change on each iteration
        print(weights)
        self.all_estimators = [] # will keep track of each stump and their corresponding weights
        for i in range(self.num_iters):
            e_i = DecisionTreeStump()
            e_i.fit(X, Y, index_to_feature_type=self,index_to_feature_type, weights=weights)
            estimator_

    def predict(self, X):
        # all_predictions  = [tree.predict(X) for tree in self.trees]
        final = np.zeros(X.shape[0])
        '''
        for preds in all_predictions:
            final += preds
        final /= self.num_trees
        final = np.around(final)
        '''
        return final
        
        
