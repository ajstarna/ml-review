from mlreview.trees.decision_tree import DecisionTreeClassifier
from mlreview.utils.featurizing import DictVectorizer
from mlreview.utils.evaluation import cross_validation

import numpy as np
from collections import defaultdict

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
x = vectorizer.transform({'outlook': 'rain', 'temperature':'mild', 'humidity': 'normal', 'wind': 'weak'})
print(f"predicting: ['rain', 'mild', 'normal', 'weak'] = {x}")
print(tree.predict(x))

cross_validation(tree, data, Y, num_folds =len(Y))



from sklearn.datasets import load_breast_cancer
d = load_breast_cancer()
tree = DecisionTreeClassifier()
index_to_feature_type = defaultdict(lambda: 'numerical')
tree.set_index_to_feature_type(index_to_feature_type)
tree.fit(d.data, d.target)
tree.print_tree()

cross_validation(tree, d.data, d.target)
