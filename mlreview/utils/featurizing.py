import numpy as np

class DictVectorizer:
    def __init__(self):
        pass

    def fit(self, X):
        self.feature_to_index = {}
        self.index_to_feature = {}
        self.index_to_feature_type = {}

        if isinstance(X, dict):
            X = [X]

        for x in X:
            for feature, val in x.items():
                if isinstance(val, str):
                    feature = f"{feature}---{val}"

                if feature in self.feature_to_index:
                    continue
                else:
                    index = len(self.feature_to_index)
                    self.feature_to_index[feature] = index
                    self.index_to_feature[index] = feature

                    if isinstance(val, str) or isinstance(val, bool):
                        # this lets us know if a value (e.g. 1) represents a categoy or a number
                        self.index_to_feature_type[index] = 'categorical'
                    else:
                        self.index_to_feature_type[index] = 'numerical'
                        

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)


    def transform(self, X):
        if isinstance(X, dict):
            X = [X]

        data = np.zeros((len(X), len(self.feature_to_index)), dtype=int)
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
