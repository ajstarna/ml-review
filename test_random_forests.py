from mlreview.trees.random_forest import RandomForestClassifier, RandomForestRegressor
from mlreview.utils.featurizing import DictVectorizer
from mlreview.utils.evaluation import cross_validation

import numpy as np
from collections import defaultdict
from sklearn.datasets import load_breast_cancer
from sklearn.datasets import load_boston



def test_random_forest_classifier_toy():
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
    forest = RandomForestClassifier()
    print(vectorizer.index_to_feature_type)

    forest.set_index_to_feature_type(index_to_feature_type=vectorizer.index_to_feature_type)
    forest.fit(data, Y)


    x = vectorizer.transform([{'outlook': 'rain', 'temperature':'mild', 
                              'humidity': 'normal', 'wind': 'weak'},
                               {'outlook': 'sunny', 'temperature':'mild', 
                                'humidity': 'normal', 'wind': 'strong'}])



    #print(f"predicting: ['rain', 'mild', 'normal', 'weak'] = {x1}")
    #x2 = vectorizer.transform({'outlook': 'sunny', 'temperature':'mild', 
    #                          'humidity': 'normal', 'wind': 'strong'})
    #print(f"predicting: ['sunny', 'mild', 'normal', 'strong'] = {x2}")
    #to_predict = np.array([x1, x2])
    #print(to_predict)

    #print(to_predict.shape)
    print(x)
    print(x.shape)
    print(forest.predict(x))

    #cross_validation(tree, data, Y, task_type='classification', num_folds =len(Y))


def test_random_forest_classifier_breast_cancer():
    # this actually takes many seconds, even though the data is small
    d = load_breast_cancer()
    forest = RandomForestClassifier(num_trees=50, max_depth=None, num_features_to_sample_from=(int(d.data.shape[1]/3)))
    index_to_feature_type = defaultdict(lambda: 'numerical')
    forest.set_index_to_feature_type(index_to_feature_type)
    #forest.fit(d.data, d.target)
    #forest.print_tree()
    cross_validation(forest, d.data, d.target, num_folds=5, task_type='classification')
    # with 5 folds, 10 trees, max_depth=None, num_features_to_sample_from = 1/3 of them (i.e. 10)
    # p=0.9743589743589743, r=0.9633802816901409, f=0.9688385269121813
    # with 5 folds, 10 trees, max_depth=None, num_features_to_sample_from = None (i.e. all 30)
    # p=0.952513966480447, r=0.9605633802816902, f=0.9565217391304348

    # yahoo! This ~confirms that sampling from a random subset of features at each node is better,
    # since it leads to trees that are less correlated

    # once more with more trees...
    # with 5 folds, [[50 trees]], max_depth=None, num_features_to_sample_from = 1/3 of them (i.e. 10)
    # p=0.9719101123595506, r=0.9774011299435028, f=0.9746478873239437
    # really good results

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
    tree = DecisionTreeRegressor(max_depth=1)
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
    tree = DecisionTreeRegressor(max_depth=3)
    index_to_feature_type = defaultdict(lambda: 'numerical')
    tree.set_index_to_feature_type(index_to_feature_type)
    tree.fit(d.data, d.target)
    tree.print_tree()
    cross_validation(tree, d.data, d.target, task_type='regression', num_folds=5)




#test_random_forest_classifier_toy()

test_random_forest_classifier_breast_cancer()

#test_decision_tree_regressor_toy()
#test_decision_tree_regressor_boston_house()
