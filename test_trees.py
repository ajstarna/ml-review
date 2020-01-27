from mlreview.trees.decision_tree import DecisionTreeClassifier
from mlreview.utils.featurizing import DictVectorizer

if __name__ == "__main__":

    '''
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
    '''

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
    print(data)
    tree = DecisionTreeClassifier()
    tree.fit(data, Y)
    tree.print_tree()
    print("predicting: {['rain', 'mild', 'normal', 'weak'])}")
    print(tree.predict(['rain', 'mild', 'normal', 'weak']))


