from mlreview.trees.decision_tree import DecisionTreeClassifier
from mlreview.utils.featurizing import DictVectorizer


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

Y = [0,0,1,1,1,0,1,0,1,1,1,1,1,0]

vectorizer = DictVectorizer()
data = vectorizer.fit_transform(X)

print(vectorizer.feature_to_index)
tree = DecisionTreeClassifier()
print(vectorizer.index_to_feature_type)
tree.fit(data, Y, index_to_feature_type=vectorizer.index_to_feature_type)
tree.print_tree()
x = vectorizer.transform({'outlook': 'rain', 'temperature':'mild', 'humidity': 'normal', 'wind': 'weak'})
print(f"predicting: ['rain', 'mild', 'normal', 'weak'] = {x}")
print(tree.predict(x))




from sklearn.datasets import load_breast_cancer
d = load_breast_cancer()
tree = DecisionTreeClassifier()
tree.fit(d.
data, d.target, index_to_feature_type=defaultdict(lambda: 'numerical'))
tree.print_tree()
#x = vectorizer.transform({'outlook': 'rain', 'temperature':'mild', 'humidity': 'normal', 'wind': 'weak'})
#print(f"predicting: ['rain', 'mild', 'normal', 'weak'] = {x}")
#print(tree.predict(x))

