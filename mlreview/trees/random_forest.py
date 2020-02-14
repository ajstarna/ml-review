'''
https://link.springer.com/content/pdf/10.1023%2FA%3A1010933404324.pdf
'''

from .decision_tree import DecisionTreeClassifier, DecisionTreeRegressor
from ..utils.bootstrapping import get_bootstrapped_data_set

from collections import defaultdict
import numpy as np

class RandomForestBase:

    def __init__(self, num_trees=10, num_features_to_sample_from=1, max_depth=None):
        self.num_trees = num_trees
        self.num_features_to_sample_from = num_features_to_sample_from
        if max_depth is None:
            max_depth = float('inf')
        self.max_depth = max_depth

    def set_index_to_feature_type(self, index_to_feature_type):
        self.index_to_feature_type = index_to_feature_type

    def fit(self, X, Y):
        self.trees = []
        self.data_index_to_trees_used = defaultdict(set) # if want to use for out-of-bag model evaluation
        for i in range(self.num_trees):
            print(f"FITTING TREE {i}")
            data_dict = get_bootstrapped_data_set(X, Y)
            data = data_dict['data']
            target = data_dict['target']
            fit_tree_i = self.DecisionTreeType(
                max_depth=self.max_depth, 
                num_features_to_sample_from=self.num_features_to_sample_from
                )
            fit_tree_i.set_index_to_feature_type(self.index_to_feature_type)
            fit_tree_i.fit(data,target)
            #fit_tree_i.print_tree()
            self.trees.append(fit_tree_i)
            indices_used = data_dict['indices_used']
            #print(f'indices used: {indices_used}')

            for index in indices_used:
                self.data_index_to_trees_used[index].add(i)


    def predict(self, X):
        all_predictions  = [tree.predict(X) for tree in self.trees]
        final = np.zeros(X.shape[0])
        for preds in all_predictions:
            final += preds
        final /= self.num_trees
        final = np.around(final)
        return final
        

class RandomForestClassifier(RandomForestBase):
    DecisionTreeType = DecisionTreeClassifier

class RandomForestRegressor(RandomForestBase):
    DecisionTreeType = DecisionTreeRegressor
