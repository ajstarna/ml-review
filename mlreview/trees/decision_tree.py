## code seems to be working for categorical features
## https://www.cise.ufl.edu/~ddd/cap6635/Fall-97/Short-papers/2.htm

import math
from collections import defaultdict
from collections import Counter


import numpy as np

############## For Classifier Tree ##############

def entropy(Y):
    counts_per_class = Counter(Y)
    entropy = 0
    for classification, count in counts_per_class.items():
        p_c = count/len(Y)
        entropy -= (p_c*math.log(p_c, 2))
    return entropy

assert(round(entropy([0,0,0,0,0,1,1,1,1,1,1,1,1,1]), 2) == 0.94)

def gain(current_entropy, Y_split, total_Y_len):
    # information gain, AKA reduction of entropy
    # we want each split to be similar to itself.
    # All 0s or all 1s is zero entropy, so the reduction will be maxium
    new_entropy = 0
    for current_split in Y_split:
        new_entropy += (len(current_split)/total_Y_len) * entropy(current_split)
    reduction = current_entropy - new_entropy
    return reduction

############## For Regressor Tree ##############

def standard_deviation(Y):
    # could just use np.std, but this is cooler
    std =  np.sqrt(np.mean((Y - Y.mean())**2))
    print(f'std = {std}')
    return std

def standard_deviation_reduction(current_standard_deviation, Y_split, total_Y_len):
    # when deciding on a split, we want the new subsets to have an overall lower standard deviation
    # i.e. within each split, the data is more similar to eachother
    new_standard_deviation = 0
    # new standard deviation is the weighted after of the separate standard deviation
    for current_split in Y_split:
        new_standard_deviation += (len(current_split)/total_Y_len) * standard_deviation(current_split)

    reduction = current_standard_deviation - new_standard_deviation # if new standard deviation is lower, then reduction is postive,
    return reduction




class Node:
    def __init__(self, feature_index):
        self.feature_index = feature_index


class TerminalNode:
    def __init__(self, return_val):
        self.return_val = return_val

    def evaluate(self, *args):
        return self.return_val

    def print_self_and_subtree(self, indent_level:int):
        tab = "\t" * indent_level
        print(f'{tab}terminal val of {self.return_val}')


'''                
class PositiveTerminalNode:
    def evaluate(self, x):
        return 1

    def print_self_and_subtree(self, indent_level:int):
        tab = "\t" * indent_level
        print(f'{tab}positive terminal')

class NegativeTerminalNode:
    def evaluate(self, x):
        return 0

    def print_self_and_subtree(self, indent_level:int):
        tab = "\t" * indent_level
        print(f'{tab}negative terminal')
'''

class CategoricalNode(Node):
    def __init__(self, feature_index, all_feature_vals=None):
        super().__init__(feature_index)
        self.all_feature_vals = all_feature_vals
        self.category_children = {}

    def evaluate(self, x):
        #print(f'evaluating index = {self.feature_index} and val = {x[self.feature_index]}')
        feature_val = x[self.feature_index]
        child = self.category_children[feature_val]
        return child.evaluate(x)

    def add_child_from_feature_key(self, feature_key, node):
        self.category_children[feature_key] = node

    def print_self_and_subtree(self, indent_level:int):
        tab = "\t" * indent_level
        print(f'{tab}feature index = {self.feature_index}') #, feature vals = {self.all_feature_vals}')
        for val, child in self.category_children.items():
            print(f'{tab}entering subtree for val = {val}')
            child.print_self_and_subtree(indent_level=indent_level+1)
        #print(f'returning from index = {self.feature_index}')

class NumericalNode(Node):
    def __init__(self, feature_index, threshold_val):
        super().__init__(feature_index)
        self.threshold_val = threshold_val
        self.less_than_child = None
        self.greater_than_child = None


    def evaluate(self, x):
        feature_val = x[self.feature_index]
        if feature_val <= self.threshold_val:
            return self.less_than_child.evaluate(x)
        else:
            return self.greater_than_child.evaluate(x)

    def add_child_from_feature_key(self, feature_key, node):
        if feature_key == 'less':
            self.less_than_child = node
        elif feature_key == 'greater':
            self.greater_than_child = node
        else:
            raise Exception("Improper feature key provided to a numerical node!")

    def print_self_and_subtree(self, indent_level:int):
        tab = "\t" * indent_level
        print(f'{tab}feature index = {self.feature_index}') #, feature vals = {self.all_feature_vals}')
        print(f'{tab}entering subtree for val <= {self.threshold_val}')
        self.less_than_child.print_self_and_subtree(indent_level=indent_level+1)
        print(f'{tab}entering subtree for val > {self.threshold_val}')
        self.greater_than_child.print_self_and_subtree(indent_level=indent_level+1)



class DecisionTreeBase:

    def __init__(self):
        pass

    def print_tree(self):
        if self.root is None:
            print('root is None')
        
            return
        self.root.print_self_and_subtree(indent_level=0)


    def set_index_to_feature_type(self, index_to_feature_type):
        self.index_to_feature_type = index_to_feature_type

    def fit(self, X, Y, index_to_feature_type=None):
        assert(len(X) == len(Y))
        self.used_indices = set()
        if index_to_feature_type is None:
            index_to_feature_type = self.index_to_feature_type

        self.root = self.recursively_build_tree(X, Y, index_to_feature_type)

    def recursively_build_tree(self, X, Y, index_to_feature_type):
        new_node, X_split, Y_split = self.find_split(X,Y, index_to_feature_type)        
        #if new_node.is_positive_terminal or new_node.is_negative_terminal:
        if isinstance(new_node, TerminalNode):
            # end once we have a terminal
            return new_node
        self.used_indices.add(new_node.feature_index)
        #print(f'using index = {new_node.feature_index}')
    
        for feature_key in X_split:
            new_node.add_child_from_feature_key(feature_key, self.recursively_build_tree(X_split[feature_key], Y_split[feature_key], index_to_feature_type))
        #for feature_val in X_split:
        #    new_node.children[feature_val] = self.recursively_build_tree(X_split[feature_val], Y_split[feature_val], index_to_feature_type)

        return new_node
        
                                           
    def split_data_for_given_categorical_feature_index(self, X, Y, feature_index):
        # given an index/feature to look at, split X and Y into groups
        # The corresponding feature is categorical, just split for each possible value 
        X_split = defaultdict(list)
        Y_split = defaultdict(list)
        for x, y in zip(X, Y):
            # get each split by feature value
            # the key is the value of feature at index, and it maps to a list of all
            # xs or ys for those corresponding values
            X_split[x[feature_index]].append(x)
            Y_split[x[feature_index]].append(y)

        node = CategoricalNode(feature_index=feature_index)#, all_feature_vals=all_feature_vals)

        for val in X_split.keys():
            X_split[val] = np.array(X_split[val])
        for val in Y_split.keys():
            Y_split[val] = np.array(Y_split[val])
        return node, X_split, Y_split


    def split_data_for_given_numerical_feature_index(self, X, Y, feature_index, current_score):
        # given an index/feature to look at, split X and Y into groups
        # The feature is numerical, look over all possible <= splits on values and return the 
        # best split
        sorted_x_y = sorted([(x, y) for x, y in zip(X,Y)], key=lambda tup: tup[0][feature_index])
        used_vals = set()
        best_improvement = float('-inf')
        for i, x_y in enumerate(sorted_x_y):
            x, y = x_y
            feature_val = x[feature_index]
            if feature_val in used_vals:
                continue
            else:
                used_vals.add(feature_val)

            split_index = i+1
            keep_extending = True
            while keep_extending:
                if split_index < len(sorted_x_y) and sorted_x_y[split_index] == feature_val:
                    split_index += 1
                else:
                    keep_extending = False

            less_Y = np.array(y for x,y in sorted_x_y[0:split_index])
            more_Y = np.array(y for x,y in sorted_x_y[split_index:])
            Y_split = (less_Y, more_Y)
            new_improvement = self.improvement_function(current_score, Y_split, len(Y))
            if new_improvement > best_improvement:
                best_improvement = new_improvement
                threshold_val = feature_val
                less_X = [x for x,y in sorted_x_y[0:split_index]]
                more_X = [x for x,y in sorted_x_y[split_index:]]
                best_Y_split = {'less': less_Y, 'greater': more_Y}
                best_X_split = {'less': less_X, 'greater': more_X}

        node = NumericalNode(feature_index=feature_index, threshold_val=threshold_val)
        return node, best_X_split, best_Y_split

                                           
    def find_split(self, X, Y, index_to_feature_type):
        sum_Y = sum(Y)
        if sum_Y == len(Y):
            #print('positive endpoint')
            return PositiveTerminalNode(), None, None
        if sum_Y == 0:
            #print('negative endpoint')
            return NegativeTerminalNode(), None, None

        if len(self.used_indices) == len(X[0]):
            # we have already split on each attribute, and this is as good as we get
            # TODO: is this actually needed? Find proof of algorithm
            if sum_Y >= (len(Y) / 2):
                print('taking best guest positive')
                return PositiveTerminalNode(), None, None
            else:
                print('taking best guest negative')
                return NegativeTerminalNode(), None, None
                                           
        current_score = self.score_function(Y)

        best_improvement = float('-inf')
        best_Y_split = None
        best_Y_split = None
        for index in range(len(X[0])):
            # each index into x corresponds to a single feature
            if index in self.used_indices:
                # can't split on the same feature again
                continue
                                           
            if index_to_feature_type[index] == 'categorical':
                node, X_split, Y_split = self.split_data_for_given_categorical_feature_index(X, Y, index)
            else:
                node, X_split, Y_split = self.split_data_for_given_numerical_feature_index(X, Y, index, current_score)

            improvement_of_split = self.improvement_function(current_score, Y_split.values(), len(Y))
            if improvement_of_split > best_improvement:
                best_improvement = improvement_of_split
                best_node = node
                best_X_split = X_split
                best_Y_split = Y_split
            else:
                pass
        return best_node, best_X_split, best_Y_split


    def predict(self, X):
        predictions = []
        for x in X:
            predictions.append(self.root.evaluate(x))
        return predictions



class DecisionTreeClassifier(DecisionTreeBase):
    def score_function(self, *args):
        return entropy(*args)

    def improvement_function(self, *args):
        return gain(*args)

class DecisionTreeRegressor(DecisionTreeBase):
    def score_function(self, *args):
        return standard_deviation(*args)

    def improvement_function(self, *args):
        return standard_deviation_reduction(*args)
