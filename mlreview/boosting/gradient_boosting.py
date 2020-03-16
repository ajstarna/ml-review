'''
https://explained.ai/gradient-boosting/index.html

https://www.youtube.com/watch?v=3CC4N4z3GJc

'''
from ..trees.decision_tree import DecisionTreeClassifier, DecisionTreeRegressor, TerminalNode
from ..utils.bootstrapping import get_bootstrapped_data_set

from collections import defaultdict
import numpy as np

class GradientBoostingRegressor:

    def __init__(self, num_estimators=10, tree_depth=3, learning_rate=0.1, subset_size=0.5, delta_stop=0.01):
        self.num_estimators = num_estimators
        self.tree_depth = tree_depth # the max depth of each of the estimator tree
        self.learning_rate = learning_rate
        self.subset_size = subset_size # the portion of the training set that each estimator uses at random
        self.delta_stop = delta_stop

    def set_index_to_feature_type(self, index_to_feature_type):
        self.index_to_feature_type = index_to_feature_type


    def fit(self, X, Y):

        print(f'X.shape = {X.shape}')
        print(f'Y.shape = {Y.shape}')

        self.all_estimators = [] # will keep track of each stump and their corresponding weights
        mean = Y.mean()
        first_estimator = TerminalNode(return_val=mean) # first predictor just gives the mean
        self.all_estimators.append((first_estimator, 1.0))
        for i in range(self.num_estimators):
            print(f'estimator {i} of {self.num_estimators}')
            current_data_subset = X # we get a random subset (without replacement) to add randomness to each estimator

            current_predictions = self.predict(current_data_subset)
            mse = np.square(np.subtract(Y, current_predictions)).mean()
            print(f'current mse = {mse}')
            # print(f'current_predictions.shape = {current_predictions.shape}')
            current_residuals = Y - current_predictions
            
            # now we want to fit a tree to the residual
            e_i = DecisionTreeRegressor(max_depth=self.tree_depth)
            e_i.fit(current_data_subset, current_residuals, index_to_feature_type=self.index_to_feature_type)
            self.all_estimators.append((e_i, self.learning_rate))

        print("done fitting:")
        #print(self.all_estimators)

    def predict(self, X):
        all_predictions  = [e_i.predict(X) * learning_rate for e_i, learning_rate in self.all_estimators]
        final = np.zeros(X.shape[0])
        for preds in all_predictions:
            final += preds
        #final = np.sum(all_predictions, axis=1)
        return final
        
        
