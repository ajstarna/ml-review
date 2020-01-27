import numpy as np

class DictVectorizer:
    def __init__(self):
        pass

    def fit(self, X):
        self.feature_to_index = {}
        self.index_to_feature = {}

        for x in X:
            for feature, val in x.items():
                if isinstance(val, str):
                    feature = f"{feature}---{val}"
                if feature in self.feature_to_index:
                    continue
                else:
                    self.feature_to_index[feature] = len(self.feature_to_index)
                    self.index_to_feature[self.feature_to_index[feature]] = feature
                                  
        
    def fit_transform(self, X):
        self.fit(X)
        data = np.zeros((len(X), len(self.feature_to_index)))
        for i, x in enumerate(X):
            for feature, val in x.items():
                if isinstance(val, str):
                    feature = f"{feature}---{val}"
                    val = 1
                index = self.feature_to_index[feature]
                data[i][index] = val
        return data




if __name__ == "__main__":

    data = [
        {'pet': 'dog', 'colour': 'brown', 'mass': 21.5},
        {'pet': 'cat', 'colour': 'brown', 'mass': 10.1},
        {'pet': 'dog', 'colour': 'white', 'mass': 25}
        ]


    vectorizer = DictVectorizer()
    fit_data = vectorizer.fit_transform(data)

    print(fit_data)
    print(vectorizer.feature_to_index)
    print(vectorizer.index_to_feature)
