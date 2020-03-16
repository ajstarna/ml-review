#!/usr/bin/env python3

from mlreview.trees.decision_tree import DecisionTreeClassifier, DecisionTreeRegressor
from mlreview.trees.random_forest import RandomForestClassifier, RandomForestRegressor
from mlreview.boosting.ada_boost import AdaBoostClassifier
from mlreview.boosting.gradient_boosting import GradientBoostingRegressor

from mlreview.utils.featurizing import DictVectorizer
from mlreview.utils.evaluation import cross_validation

import numpy as np
from collections import defaultdict
from sklearn.datasets import load_breast_cancer
from sklearn.datasets import load_boston
import argparse


def test_classifier_toy(classifier):
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
    print(vectorizer.index_to_feature_type)

    try:
        classifier.set_index_to_feature_type(index_to_feature_type=vectorizer.index_to_feature_type)
    except:
        # kinda gross. i wanna refactor this to not need the vectorizer to pass in this info.
        # just let the fit methods figure this out. less work then for the general API
        pass

    #classifier.fit(data, Y)
    #classifier.print_tree()
    x = vectorizer.transform({'outlook': 'rain', 'temperature':'mild', 
                              'humidity': 'normal', 'wind': 'weak'})
    #print(f"predicting: ['rain', 'mild', 'normal', 'weak'] = {x}")
    #print(classifier.predict(x))

    cross_validation(classifier, data, Y, task_type='classification', num_folds=3)



def test_classifier_breast_cancer(classifier):
    # this actually takes many seconds, even though the data is small
    d = load_breast_cancer()
    index_to_feature_type = defaultdict(lambda: 'numerical')
    classifier.set_index_to_feature_type(index_to_feature_type)
    #classifier.fit(d.data, d.target)
    print(d.data.shape)
    cross_validation(classifier, d.data, d.target, task_type='classification', num_folds=3)
    # tree: p=0.9371428571428572, r=0.923943661971831, f=0.9304964539007092

    # ada boost: for 4 folds and 4 estimators:
    # p=0.9305555555555556, r=0.938375350140056, f=0.9344490934449093

    # ada boost: for 5 folds and 10 estimators
    # p=0.9303621169916435, r=0.9461756373937678, f=0.9382022471910112
    # most of the stage values are near 0. Is there a problem with the 0 versus -1 distinction?
    # or everything fine hmm?

def test_regressor_toy(regressor):
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
    print(vectorizer.index_to_feature_type)
    regressor.set_index_to_feature_type(index_to_feature_type=vectorizer.index_to_feature_type)


    regressor.fit(data, Y)
    #tree.print_tree()
    #x = vectorizer.transform({'outlook': 'rain', 'temperature':'mild', 
    #                          'humidity': 'normal', 'wind': False})
    #print(f"predicting: ['rain', 'mild', 'normal', False] = {x}")

    cross_validation(regressor, data, Y, task_type='regression', num_folds =len(Y))


def test_regressor_boston_house(regressor):
    # this actually takes many seconds, even though the data is small
    d = load_boston()
    index_to_feature_type = defaultdict(lambda: 'numerical')
    regressor.set_index_to_feature_type(index_to_feature_type)
    #regressor.fit(d.data, d.target)
    cross_validation(regressor, d.data, d.target, task_type='regression', num_folds=5)

    # decision tree no max depth --> MSE 23.56
    # decision tree 10 max depth --> MSE 23.00
    # decision tree 5 max depth --> MSE 19.2
    # decision tree 3 max depth --> MSE 18.6
    # decision tree 2 max depth --> MSE 21.07
    # decision tree 1 max depth --> MSE 28.99

    # random forest 10 trees, no max depth, and default=1/3 features to sample --> MSE 17.17
    # random forest 30 trees, 5 max depth, and default=1/3 features to sample --> MSE 15.86
    # random forest 30 trees, 3 max depth, and default=1/3 features to sample --> MSE 16.28
    # random forest 30 trees, 2 max depth, and default=1/3 features to sample --> MSE 21.4



if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--estimator', '-e', dest='estimator', required=True, 
                        help='the type of estimator to use'
                        )
                        
    parser.add_argument('--test_set', '-t', dest='test_set', default='all',
                        help='the test set to use',
                        )
                        
    args = parser.parse_args()

    if args.estimator == 'tree':
        classifier = DecisionTreeClassifier()
        regressor = DecisionTreeRegressor(max_depth=1) 
    elif args.estimator in ['random_forest', 'rf']:
        classifier = RandomForestClassifier()
        regressor = RandomForestRegressor(num_trees=30, max_depth=2)
    elif args.estimator in ['linear_regression', 'lr']:
        pass
    elif args.estimator in ['ada_boost', 'ab']:
        classifier = AdaBoostClassifier(num_estimators=25)
    elif args.estimator in ['grad_boost', 'gb']:
        # classifier = GradientBoostingClassifier(num_estimators=25)
        regressor = GradientBoostingRegressor(num_estimators=30)
    else:
        print('Unknown estimator!')
        exit()
    
    

    if args.test_set in ['ct', 'all']:
        print('Toy classifier test')
        test_classifier_toy(classifier)
    elif args.test_set in ['cb', 'all']:
        print('Cancer classifier test')
        test_classifier_breast_cancer(classifier)
    elif args.test_set in ['rt', 'all']:
        print('Toy reggressor test')
        test_regressor_toy(regressor)
    elif args.test_set in ['rb', 'all']:
        print('Boston house regressor test')
        test_regressor_boston_house(regressor)
    else:
        print('Unknown test set')

