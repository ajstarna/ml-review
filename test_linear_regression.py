from mlreview.linear_regression.linear_regression import LinearRegressor
from mlreview.utils.evaluation import cross_validation

import numpy as np
from collections import defaultdict
from sklearn.datasets import load_breast_cancer
from sklearn.datasets import load_boston


from sklearn.linear_model import LinearRegression

def test_linear_regressor_boston_house():
    # this actually takes many seconds, even though the data is small
    d = load_boston()
    print(d)
    regressor = LinearRegressor(max_iters=1000)
    regressor.fit(d.data, d.target)
    #print(regressor.b)
    #print(regressor.thetas)

    regression = LinearRegression()
    regression.fit(d.data, d.target)
    print(regression.intercept_)
    print(regression.coef_)
    cross_validation(regressor, d.data, d.target, task_type='regression', num_folds=10)
    # MSE of 94 atm. Is there a bug?


def test_linear_regressor_toy():
    X = np.array([[2, 10],  [5,19], [7,12], [9, 22]])
    print(X.shape)
    Y = np.array([[3.2, 5.7, 8, 9.9]]).T # .T to force column vector
    print(Y.shape)
    regressor = LinearRegressor(max_iters=100)
    regressor.fit(X, Y)
    regressor.predict(X)
    


test_linear_regressor_boston_house()
#test_linear_regressor_toy()
