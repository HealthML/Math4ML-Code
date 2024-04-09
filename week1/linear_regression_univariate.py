import numpy as np


class UnivariateLinearRegression:
    def __init__(self):
        self.x = None
        self.y = None
        self.w = None
        self.b = None
        
    def train(self, x, y):
        self.x = x
        self.y = y
        self.w = np.sum((x - np.mean(x)) * (y - np.mean(y))) / np.sum((x - np.mean(x))**2)
        self.b = np.mean(y) - self.w * np.mean(x)

    def pred(self, x):
        y = self.w * x + self.b
        return y
    
    def mse(self, x=None, y=None):
        if x is None:
            x = self.x
        if y is None:
            y = self.y
        y_pred = self.pred(x)
        mse = np.mean((y - y_pred)**2)
        return mse
    
    def score(self, x=None, y=None):
        return -self.mse(x, y)

