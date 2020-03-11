'''

'''
from ..trees.decision_tree import DecisionTreeClassifier, DecisionTreeRegressor, TerminalNode
from ..utils.bootstrapping import get_bootstrapped_data_set

from collections import defaultdict
import numpy as np

class GradientBoostingRegressor:

    def __init__(self, num_estimators=10, tree_depth=4, learning_rate=0.1, subset_size=0.5, delta_stop=0.01):
        self.num_estimators = num_estimators
        self.tree_depth = tree_depth # the max depth of each of the estimator tree
        self.learning_rate = learning_rate
        self.subset_size = subset_size # the portion of the training set that each estimator uses at random
        self.delta_stop = delta_stop

    def set_index_to_feature_type(self, index_to_feature_type):
        self.index_to_feature_type = index_to_feature_type


    def fit(self, X, Y):
        self.all_estimators = [] # will keep track of each stump and their corresponding weights
        mean = Y.mean()
        first_estimator = TerminalNode(return_val=mean) # first predictor just gives the mean
        self.all_estimators.append(first_estimator)
        for i in range(self.num_estimators):
            print(f'estimator {i} of {self.num_estimators}')
            current_data_subset = X # we get a random subset (without replacement) to add randomness to each estimator

            current_predictions = self.predict(current_data_set)
            current_residuals = Y - current_predictions
            
            # now we want to fit a tree to the residual
            e_i = DecisionTreeRegressor().fit(current_data_set, current_residuals)
            self.all_estimators.append

        print("done fitting:")
        print(self.all_estimators)

    def predict(self, X):
        final = np.zeros(X.shape[0])
        for preds in all_predictions:
            final += preds
        final[final > 0] = 1
        final[final < 0] = 0
        return final
        
        
