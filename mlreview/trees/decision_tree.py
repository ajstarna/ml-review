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
    #print(f'std = {std}')
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

############## For Decision Stump ##############

def weighted_error(Y, preds, weights):
    misclassifications = preds != Y
    # we get the weights of data that we incorrectly labelled
    misclassification_weights = misclassifications * weights
    # error is relative to the weights of each data
    error =  np.sum(misclassification_weights) / np.sum(weights)
    return error


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


class CategoricalNode(Node):
    def __init__(self, feature_index, all_feature_vals=None):
        super().__init__(feature_index)
        self.all_feature_vals = all_feature_vals
        self.category_children = {}

    def evaluate(self, x):
        #print(f'evaluating index = {self.feature_index} and val = {x[self.feature_index]}')
        feature_val = x[self.feature_index]
        try:
            child = self.category_children[feature_val]
        except Exception as e:
            print(f'self.feature_index = {self.feature_index}')
            print(f'x = {x}')
            print(f'self.category_children = {self.category_children}')
            print(e)
            exit(-1)
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

    def __init__(self, max_depth=None, num_features_to_sample_from=None):
        if max_depth is None:
            max_depth = float('inf')
        self.max_depth = max_depth
        # for random forests; we may not want to look at all the features
        # when we create a node
        self.num_features_to_sample_from = num_features_to_sample_from


    def print_tree(self):
        if self.root is None:
            print('root is None')
        
            return
        self.root.print_self_and_subtree(indent_level=0)

    def set_index_to_feature_type(self, index_to_feature_type):
        self.index_to_feature_type = index_to_feature_type

    def feature_indices_to_sample(self, total_num_features):
        if self.num_features_to_sample_from is None:
            num_features_to_sample_from = total_num_features
            #num_features_to_sample = int(math.log(num_features, 2)))
        else:
            num_features_to_sample_from = self.num_features_to_sample_from

        #print(f'num features to sample from = {num_features_to_sample_from}')
        #print(f'total num features = {total_num_features}')
        sample_indices = np.random.choice(range(total_num_features), num_features_to_sample_from, replace=False)
        #print(f'sample_indices = {sample_indices}')
        return sample_indices

    def fit(self, X, Y, index_to_feature_type=None):
        '''
        print()
        print()
        print()
        print('=====================================')
        print("ABOUT TO FIT")
        print('=====================================')
        print()
        print()
        '''
        assert(len(X) == len(Y))
        #self.used_indices = set()
        if index_to_feature_type is None:
            index_to_feature_type = self.index_to_feature_type

        self.root = self.recursively_build_tree(X, Y, depth=0, index_to_feature_type=index_to_feature_type)

    def recursively_build_tree(self, X, Y, depth, index_to_feature_type):
        new_node, X_split, Y_split = self.find_split(X, Y, depth, index_to_feature_type)        
        depth += 1
        if isinstance(new_node, TerminalNode):
            # end once we have a terminal
            return new_node
    
        for feature_key in X_split:
            new_node.add_child_from_feature_key(feature_key, 
                                                self.recursively_build_tree(X_split[feature_key], Y_split[feature_key], 
                                                                            depth, index_to_feature_type))

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


    def generate_numerical_splits(self, X, Y, feature_index, weights=None):
        # For a given feature index, this generator yeilds all the possible splits based on x values
        # at that given index. 
        # The less_Xs all have a feature value less than OR EQUAL TO the returned feature_val, 
        # while the more_Xs all have a feature val greater than

        if weights is not None:
            # getting a little gross with passing weights in now way after the fact...
            sorted_data = sorted([(x, y, w) for x, y, w in zip(X,Y,weights)], key=lambda tup: tup[0][feature_index])
        else:
            sorted_data = sorted([(x, y) for x, y in zip(X,Y)], key=lambda tup: tup[0][feature_index])
        used_vals = set()
        for i, data in enumerate(sorted_data):
            if weights is not None:
                x, y, w = data
            else:
                x, y = data
            feature_val = x[feature_index]
            if feature_val in used_vals:
                continue
            else:
                used_vals.add(feature_val)

            split_index = i + 1
            keep_extending = True
            while keep_extending:
                if split_index < len(sorted_data) and sorted_data[split_index] == feature_val:
                    split_index += 1
                else:
                    keep_extending = False

            if weights is not None:
                # ahhh so gross... what have i done!
                less_Y = np.array([y for x,y,w in sorted_data[0:split_index]])
                more_Y = np.array([y for x,y,w in sorted_data[split_index:]])
                less_X = np.array([x for x,y,w in sorted_data[0:split_index]])
                more_X = np.array([x for x,y,w in sorted_data[split_index:]])
                less_weights = np.array([w for x,y,w in sorted_data[0:split_index]])
                more_weights = np.array([w for x,y,w in sorted_data[split_index:]])
                yield less_X, less_Y, more_X, more_Y, feature_val, less_weights, more_weights
            else:
                less_Y = np.array([y for x,y in sorted_data[0:split_index]])
                more_Y = np.array([y for x,y in sorted_data[split_index:]])
                less_X = np.array([x for x,y in sorted_data[0:split_index]])
                more_X = np.array([x for x,y in sorted_data[split_index:]])
                yield less_X, less_Y, more_X, more_Y, feature_val


    def split_data_for_given_numerical_feature_index(self, X, Y, feature_index, current_score):
        # given an index/feature to look at, split X and Y into groups
        # The feature is numerical, look over all possible <= splits on values and return the 
        # best split
        best_improvement = float('-inf')
        for split_return in self.generate_numerical_splits(X, Y, feature_index):
            less_X, less_Y, more_X, more_Y, feature_val = split_return
            Y_split = (less_Y, more_Y)
            new_improvement = self.improvement_function(current_score, Y_split, len(Y))
            if new_improvement > best_improvement:
                best_improvement = new_improvement
                threshold_val = feature_val
                best_Y_split = {'less': less_Y, 'greater': more_Y}
                best_X_split = {'less': less_X, 'greater': more_X}

        node = NumericalNode(feature_index=feature_index, threshold_val=threshold_val)
        return node, best_X_split, best_Y_split


    def possibly_create_terminal_node(self, X, Y, *args):
        # the sub-classes implement this
        raise NotImplementedError()

                                           
    def find_split(self, X, Y, depth, index_to_feature_type):                   
        current_score = self.score_function(Y)
        terminal_node = self.possibly_create_terminal_node(X, Y, depth, current_score)
        if terminal_node is not None:
            # if we are at the end, for any condition, then terminal_node will be non-None,
            #print(f'terminal node with val of {terminal_node.return_val}')
            return terminal_node, None, None

        best_improvement = 0
        best_X_split = None
        best_Y_split = None
        for index in self.feature_indices_to_sample(total_num_features=X.shape[1]):
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

        if best_Y_split is None:
            # if couldn't actually find any improvement
            # Note: is this possible, or just a sanity check hmm?
            print("NO BETTER SPLIT SO LETS END ON TERMINAL")
            terminal_node = self.create_terminal_node(Y)
            return terminal_node, None, None

        return best_node, best_X_split, best_Y_split


    def predict(self, X):
        predictions = []
        for x in X:
            predictions.append(self.root.evaluate(x))
        return np.array(predictions)



class DecisionTreeClassifier(DecisionTreeBase):
    def score_function(self, *args):
        return entropy(*args)

    def improvement_function(self, *args):
        return gain(*args)

    def create_terminal_node(self, Y, mean=None):
        if mean is None:
            mean = Y.mean()
        return TerminalNode(return_val=round(mean))

    def possibly_create_terminal_node(self, X, Y, depth, *args):
        if depth > self.max_depth:
            return self.create_terminal_node(Y)
        sum_Y = sum(Y)
        if sum_Y == len(Y):
            #print('positive endpoint')
            #return PositiveTerminalNode(), None, None
            return TerminalNode(return_val=1)
        if sum_Y == 0:
            #print('negative endpoint')
            return TerminalNode(return_val=0)

        return None

class DecisionTreeRegressor(DecisionTreeBase):

    def __init__(self, coefficient_of_deviation_threshold=0.1, **kwargs):
        print('kwargs')
        print(kwargs)
        super().__init__(**kwargs)
        self.coefficient_of_deviation_threshold = coefficient_of_deviation_threshold

    def score_function(self, *args):
        return standard_deviation(*args)

    def improvement_function(self, *args):
        return standard_deviation_reduction(*args)

    def create_terminal_node(self, Y, mean=None):
        if mean is None:
            mean = Y.mean()
        return TerminalNode(return_val=mean)

    def possibly_create_terminal_node(self, X, Y, depth, current_standard_deviation):
        mean = Y.mean()
        if current_standard_deviation / mean < self.coefficient_of_deviation_threshold \
                or depth > self.max_depth:
            #or len(self.used_indices) == len(X[0]):
            # our coefficient of deviation is low enough that we hit the threshold OR
            # we have already split on each attribute, and this is as good as we get
            return self.create_terminal_node(Y, mean=mean)
        return None

class DecisionTreeStump(DecisionTreeBase):

    def __init__(self, max_depth=None):
        super().__init__(max_depth=1)


                             
    def score_function(self, *args):
        return entropy(*args)

    def improvement_function(self, *args):
        return gain(*args)

    def create_terminal_node(self, Y, mean=None):
        if mean is None:
            mean = Y.mean()
        val = round(mean)
        #if val == 0:
        #    val = -1 # for estimator weighting in ada boost
        return TerminalNode(return_val=val)



    def split_data_for_given_categorical_feature_index(self, X, Y, feature_index, weights):
        # given an index/feature to look at, split X and Y into groups
        # The corresponding feature is categorical, just split for each possible value 
        X_split = defaultdict(list)
        Y_split = defaultdict(list)
        weights_split = defaultdict(list)
        for x, y in zip(X, Y):
            # get each split by feature value
            # the key is the value of feature at index, and it maps to a list of all
            # xs or ys for those corresponding values
            X_split[x[feature_index]].append(x)
            Y_split[x[feature_index]].append(y)
            weights_split[x[feature_index]].append(y)

        node = CategoricalNode(feature_index=feature_index)#, all_feature_vals=all_feature_vals)

        for val in X_split.keys():
            X_split[val] = np.array(X_split[val])
        for val in Y_split.keys():
            Y_split[val] = np.array(Y_split[val])
        return node, X_split, Y_split, weights_split



    def get_preds_by_common(self, Y):
        # just returns the majority classification as an array
        if np.sum(Y) > (Y.shape[0] / 2):
            return np.ones(Y.shape[0])
        else:
            return np.zeros(Y.shape[0])

    def get_error_from_split(self, Y_split, weights_split):
        all_Ys = []
        all_preds = []
        all_weights = []
        for name in Y_split:
            Y = Y_split[name]
            all_Ys.append(Y)

            weights = weights_split[name]
            all_weights.append(weights)

            preds = self.get_preds_by_common(Y)

            all_preds.append(preds)

        total_Y = np.concatenate(all_Ys)
        total_preds = np.concatenate(all_preds)
        total_weights = np.concatenate(all_weights)
        error = weighted_error(total_Y, total_preds, total_weights)
        return error

    def split_data_for_given_numerical_feature_index(self, X, Y, feature_index, weights):
        # given an index/feature to look at, split X and Y into groups
        # The feature is numerical, look over all possible <= splits on values and return the 
        # best split
        best_error = float('inf')
        #print(f'about to gen on feature index {feature_index}')
        for split_return in self.generate_numerical_splits(X, Y, feature_index, weights):
            less_X, less_Y, more_X, more_Y, feature_val, less_weights, more_weights = split_return
            X_split = {'less': less_X, 'greater': more_X}
            Y_split = {'less': less_Y, 'greater': more_Y}
            weights_split = {'less': less_weights, 'greater': more_weights}
            error = self.get_error_from_split(Y_split, weights_split)
            if error < best_error:
                #print('new best')
                #print(f'error = {error}')
                best_error = error
                threshold_val = feature_val
                best_Y_split = Y_split
                best_X_split = X_split
        node = NumericalNode(feature_index=feature_index, threshold_val=threshold_val)
        return node, best_X_split, best_Y_split, best_error
                                                                        
    def fit(self, X, Y, index_to_feature_type=None, weights=None):
        self.weights = weights # sorta hacky but dont want to refactor more of base tree
        super().fit(X, Y, index_to_feature_type=index_to_feature_type)


    def possibly_create_terminal_node(self, X, Y, depth, *args):
        if depth > self.max_depth:
            return self.create_terminal_node(Y)
        sum_Y = sum(Y)
        if sum_Y == len(Y):
            #print('positive endpoint')
            #return PositiveTerminalNode(), None, None
            return TerminalNode(return_val=1)
        if sum_Y == 0:
            #print('negative endpoint')
            return TerminalNode(return_val=0)

        return None


    def find_split(self, X, Y, depth, index_to_feature_type):                   
        terminal_node = self.possibly_create_terminal_node(X, Y, depth)

        if terminal_node is not None:
            # if we are at the end, for any condition, then terminal_node will be non-None,
            #print(f'terminal node with val of {terminal_node.return_val}')
            return terminal_node, None, None

        best_error = float('inf')
        for index in self.feature_indices_to_sample(total_num_features=X.shape[1]):
            if index_to_feature_type[index] == 'categorical':
                node, X_split, Y_split, weights_split = self.split_data_for_given_categorical_feature_index(X, Y, index, self.weights)
                error = self.get_error_from_split(Y_split, weights_split)
            else:
                node, X_split, Y_split, error = self.split_data_for_given_numerical_feature_index(X, Y, index, self.weights)

            if error < best_error:
                best_error = error
                best_node = node
                best_X_split = X_split
                best_Y_split = Y_split


        return best_node, best_X_split, best_Y_split

