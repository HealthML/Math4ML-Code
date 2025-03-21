import numpy as np
from linear_regression import OLS

class PolynomialRegression:

    @staticmethod
    def polynomial_design(x, degree=1):
        n = x.shape[0]
        X = np.ones((n,degree+1))
        for k in range(1,degree+1):
            X[:,k] = X[:,k-1] * x   
        return X

    def __init__(self, degree=1, linear_regression=None):
        if linear_regression is None:
            linear_regression=OLS()
        self.linear_regression = linear_regression
        self.degree = degree
        self.x = None
        
    def fit(self, x, y):
        self.x = x
        X = PolynomialRegression.polynomial_design(x,degree=self.degree)
        self.linear_regression.fit(x=X,y=y)

    def pred(self, x):
        X = PolynomialRegression.polynomial_design(x,degree=self.degree)
        return self.linear_regression.pred(x=X)