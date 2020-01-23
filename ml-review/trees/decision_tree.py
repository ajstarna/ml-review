import math
from collections import defaultdict
from collections import Counter

import random


def entropy(y):
    counts_per_class = Counter(y)
    entropy = 0
    for classification, count in counts_per_class.items():
        p_c = count/len(y)
        entropy -= (p_c*math.log(p_c, 2))
    return entropy

assert(round(entropy([0,0,0,0,0,1,1,1,1,1,1,1,1,1]), 2) == 0.94)

def gain(current_entropy, y_split, total_y_len):
    gain_val = current_entropy
    for current_split in y_split:
        gain_val -= (len(current_split)/total_y_len) * entropy(current_split)
    #return gain_val
    return random.randint(0,100)


class Node:
    def __init__(self, feature_index=None, feature_vals=None, is_positive_terminal=False, is_negative_terminal=False):
        self.feature_index = feature_index
        self.feature_vals = feature_vals
        self.children = {}
        self.is_positive_terminal = is_positive_terminal
        self.is_negative_terminal = is_negative_terminal

    def print_self_and_subtree(self, indent_level:int):
        tab = "\t" * indent_level
        if self.is_positive_terminal:
            print(f'{tab}positive terminal')
        elif self.is_negative_terminal:
            print(f'{tab}negative terminal')
        else:
            print(f'{tab}feature index = {self.feature_index}, feature vals = {self.feature_vals}')
            for val, child in self.children.items():
                print(f'{tab}entering subtree for val = {val}')
                child.print_self_and_subtree(indent_level=indent_level+1)
            #print(f'returning from index = {self.feature_index}')

    def evaluate(self, x):
        if self.is_positive_terminal:
            return 1
        elif self.is_negative_terminal:
            return 0
        print(f'evaluating: feature val = {x[self.feature_index]}')
        child = self.children[x[self.feature_index]]
        return child.evaluate(x)

class DecisionTree:

    def __init__(self):
        pass


    def print_tree(self):
        if self.root is None:
            print('root is None')

            return
        self.root.print_self_and_subtree(indent_level=0)

    def fit(self, X, y):
        assert(len(X) == len(y))
        self.used_indices = set()
        self.root = self.recursively_build_tree(X, y)


    def recursively_build_tree(self, X, y):
        new_node, X_split, y_split = self.find_split(X,y)        
        if new_node.is_positive_terminal or new_node.is_negative_terminal:
            # end once we have a terminal
            return new_node
        self.used_indices.add(new_node.feature_index)
        print(f'using index = {new_node.feature_index}')
        for feature_val in X_split:
            new_node.children[feature_val] = self.recursively_build_tree(X_split[feature_val], y_split[feature_val])

        return new_node
        

    def split_data_for_given_feature_index(self, X, y, index):
        x_split = defaultdict(list)
        y_split = defaultdict(list)
        for x, target in zip(X, y):
            # get each split by feature value
            # the key is the value of feature at index, and it maps to a list of all
            # xs or ys for those corresponding values
            #print(f'current x = {x}')
            #print(f'current target = {target}')
            x_split[x[index]].append(x)
            y_split[x[index]].append(target)
        return x_split, y_split


    def find_split(self, X, y):
        print('entering find split:')
        print(X)
        print(y)
        sum_y = sum(y)
        if sum_y == len(y):
            print('positive endpoint')
            return Node(is_positive_terminal=True), None, None
        if sum_y == 0:
            print('negative endpoint')
            return Node(is_negative_terminal=True), None, None

        if len(self.used_indices) == len(X[0]):
            # we have already split on each attribute, and this is as good as we get
            # TODO: is this actually needed? Find proof of algorithm
            if sum_y >= (len(y) / 2):
                print('taking best guest positive')
                return Node(is_positive_terminal=True), None, None
            else:
                print('taking best guest negative')
                return Node(is_negative_terminal=True), None, None

        current_entropy = entropy(y)
        best_gain = -1
        best_x_split = None
        best_y_split = None
        for index in range(len(X[0])):
            # each index into x corresponds to a single feature
            if index in self.used_indices:
                # can't split on the same feature again
                continue
            x_split, y_split = self.split_data_for_given_feature_index(X, y, index)
            #print()
            #print(f'splitting on index = {index}')
            #print(f'x_split = {x_split}')
            #print(f'y_split = {y_split}')
            all_feature_vals = list(x_split.keys())
            gain_of_split = gain(current_entropy, y_split.values(), len(y))
            if gain_of_split > best_gain:
                #print(f'gain of {gain_of_split} new best')
                #print(f'index = {index}')
                best_gain = gain_of_split
                best_node = Node(feature_index=index, feature_vals=all_feature_vals)
                best_x_split = x_split
                best_y_split = y_split
            else:
                #print(f'gain of {gain_of_split} not good enough')
                #print(f'index = {index}, feature_vals = {all_feature_vals}')
                pass
        return best_node, best_x_split, best_y_split


    def predict(self, x):
        return self.root.evaluate(x)


if __name__ == "__main__":
    X = [
        ['sunny', 'hot', 'high', 'weak'],
        ['sunny', 'hot', 'high', 'strong'],
        ['overcast', 'hot', 'high', 'weak'],
        ['rain', 'mild', 'high', 'weak'],
        ['rain', 'cool', 'normal', 'weak'],
        ['rain', 'cool', 'normal', 'strong'],
        ['overcast', 'cool', 'normal', 'strong'],
        ['sunny', 'mild', 'high', 'weak'],
        ['sunny', 'cool', 'normal', 'weak'],
        ['rain', 'mild', 'normal', 'weak'],
        ['sunny', 'mild', 'normal', 'strong'],
        ['overcast', 'mild', 'high', 'strong'],
        ['overcast', 'hot', 'normal', 'weak'],
        ['rain', 'mild', 'high', 'strong'],
        ]
    y = [0,0,1,1,1,0,1,0,1,1,1,1,1,0]
    
    random.seed()
    tree = DecisionTree()
    tree.fit(X, y)
    tree.print_tree()
    print("predicting: {['rain', 'mild', 'normal', 'weak'])}")
    print(tree.predict(['rain', 'mild', 'normal', 'weak']))


    ## code seems to be working for categorical features
    ## https://www.cise.ufl.edu/~ddd/cap6635/Fall-97/Short-papers/2.htm
    ## TODO: extend to real-numbered variables
    
