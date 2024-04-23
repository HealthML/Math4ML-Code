import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

class OLS:
    def __init__(self):
        self.w = None
        self.x = None
        self.y = None
        pass
        
    def fit(self, x, y):
        self.x = x
        self.y = y
        self.w = OLS.ols(x=x, y=y)
        pass

    @staticmethod
    def ols(x, y):
        XX = x.T.dot(x)
        Xy = x.T.dot(y)
        w = np.linalg.solve(XX,Xy)
        # solving the linear equation system is equivalent to copmuting XXi.dot(Xy) as follows:
        # XXi = np.inv(XX)
        # w = XXi.dot(Xy)
        return w
    
    def pred(self, x):
        return x.dot(self.w)

    def mse(self, x=None, y=None):
        if x is None:
            x = self.x
        if y is None:
            y = self.y
        y_pred = self.pred(x)
        residuals = y - y_pred
        mse = np.mean(residuals*residuals)
        return mse
    
    def p_norm_loss(self, x=None, y=None, p=2):
        if x is None:
            x = self.x
        if y is None:
            y = self.y
        y_pred = self.pred(x)
        residuals = y - y_pred
        mse = np.mean(np.absolute(residuals).power(p))
        return mse

    def score(self, x=None, y=None):
        return -self.mse(x=x, y=y)
