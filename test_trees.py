from mlreview.trees.decision_tree import DecisionTreeClassifier
from mlreview.trees.decision_tree import DecisionTreeRegressor
from mlreview.utils.featurizing import DictVectorizer
from mlreview.utils.evaluation import cross_validation

import numpy as np
from collections import defaultdict
from sklearn.datasets import load_breast_cancer
from sklearn.datasets import load_boston



def test_decision_tree_classifier_toy():
    # decision tree people really want to know if they should play tennis
    X = [
        {'outlook': 'sunny', 'temperature': 'hot', 'humidity': 'high', 'wind': 'weak'},
        {'outlook': 'sunny', 'temperature': 'hot', 'humidity': 'high', 'wind': 'strong'},
        {'outlook': 'overcast', 'temperature': 'hot', 'humidity': 'high', 'wind': 'weak'},
        {'outlook': 'rain', 'temperature':'mild', 'humidity': 'high', 'wind': 'weak'},
        {'outlook': 'rain', 'temperature':'cool', 'humidity': 'normal', 'wind': 'weak'},
        {'outlook': 'rain', 'temperature':'cool', 'humidity': 'normal', 'wind': 'strong'},
        {'outlook': 'overcast', 'temperature':'cool', 'humidity': 'normal', 'wind': 'strong'},
        {'outlook': 'sunny', 'temperature':'mild', 'humidity': 'high', 'wind': 'weak'},
        {'outlook': 'sunny', 'temperature':'cool', 'humidity': 'normal', 'wind': 'weak'},
        {'outlook': 'rain', 'temperature':'mild', 'humidity': 'normal', 'wind': 'weak'},
        {'outlook': 'sunny', 'temperature':'mild', 'humidity': 'normal', 'wind': 'strong'},
        {'outlook': 'overcast', 'temperature':'mild', 'humidity': 'high', 'wind': 'strong'},
        {'outlook': 'overcast', 'temperature':'hot', 'humidity': 'normal', 'wind': 'weak'},
        {'outlook': 'rain', 'temperature':'mild', 'humidity': 'high', 'wind': 'strong'},
        ]

    Y = np.array([0,0,1,1,1,0,1,0,1,1,1,1,1,0])

    vectorizer = DictVectorizer()
    data = vectorizer.fit_transform(X)

    print(vectorizer.feature_to_index)
    tree = DecisionTreeClassifier()
    print(vectorizer.index_to_feature_type)


    tree.set_index_to_feature_type(index_to_feature_type=vectorizer.index_to_feature_type)
    tree.fit(data, Y)
    tree.print_tree()
    x = vectorizer.transform({'outlook': 'rain', 'temperature':'mild', 
                              'humidity': 'normal', 'wind': 'weak'})
    print(f"predicting: ['rain', 'mild', 'normal', 'weak'] = {x}")
    print(tree.predict(x))

    cross_validation(tree, data, Y, task_type='classification', num_folds =len(Y))


def test_decision_tree_classifier_breast_cancer():
    # this actually takes many seconds, even though the data is small
    d = load_breast_cancer()
    tree = DecisionTreeClassifier()
    index_to_feature_type = defaultdict(lambda: 'numerical')
    tree.set_index_to_feature_type(index_to_feature_type)
    tree.fit(d.data, d.target)
    tree.print_tree()
    cross_validation(tree, d.data, d.target, task_type='classification')




def test_decision_tree_regressor_toy():
    X = [
        {'outlook': 'rain', 'temperature': 'hot', 'humidity': 'high', 'wind': False},
        {'outlook': 'rain', 'temperature': 'hot', 'humidity': 'high', 'wind': True},
        {'outlook': 'overcast', 'temperature': 'hot', 'humidity': 'high', 'wind': False},
        {'outlook': 'sunny', 'temperature':'mild', 'humidity': 'high', 'wind': False},
        {'outlook': 'sunny', 'temperature':'cool', 'humidity': 'normal', 'wind': False},
        {'outlook': 'sunny', 'temperature':'cool', 'humidity': 'normal', 'wind': True},
        {'outlook': 'overcast', 'temperature':'cool', 'humidity': 'normal', 'wind': True},
        {'outlook': 'rain', 'temperature':'mild', 'humidity': 'high', 'wind': False},
        {'outlook': 'rain', 'temperature':'cool', 'humidity': 'normal', 'wind': False},
        {'outlook': 'sunny', 'temperature':'mild', 'humidity': 'normal', 'wind': False},
        {'outlook': 'rain', 'temperature':'mild', 'humidity': 'normal', 'wind': True},
        {'outlook': 'overcast', 'temperature':'mild', 'humidity': 'high', 'wind': True},
        {'outlook': 'overcast', 'temperature':'hot', 'humidity': 'normal', 'wind': False},
        {'outlook': 'sunny', 'temperature':'mild', 'humidity': 'high', 'wind': True},
        ]

    Y = np.array([26,30,48,46,62,23,43,36,38,48,48,62,44,30])

    vectorizer = DictVectorizer()
    data = vectorizer.fit_transform(X)

    print(vectorizer.feature_to_index)
    tree = DecisionTreeRegressor()
    print(vectorizer.index_to_feature_type)
    tree.set_index_to_feature_type(index_to_feature_type=vectorizer.index_to_feature_type)


    tree.fit(data, Y)
    tree.print_tree()
    #x = vectorizer.transform({'outlook': 'rain', 'temperature':'mild', 
    #                          'humidity': 'normal', 'wind': False})
    #print(f"predicting: ['rain', 'mild', 'normal', False] = {x}")


    #print(tree.predict(x))
    print("NOW CROSS VAL")
    cross_validation(tree, data, Y, task_type='regression', num_folds =len(Y))


def test_decision_tree_regressor_boston_house():
    # this actually takes many seconds, even though the data is small
    d = load_boston()
    print(d)
    tree = DecisionTreeRegressor()
    index_to_feature_type = defaultdict(lambda: 'numerical')
    tree.set_index_to_feature_type(index_to_feature_type)
    tree.fit(d.data, d.target)
    tree.print_tree()
    cross_validation(tree, d.data, d.target, task_type='regression', num_folds=10)




#test_decision_tree_classifier_toy()

test_decision_tree_classifier_breast_cancer()

#test_decision_tree_regressor_toy()
#test_decision_tree_regressor_boston_house()
