import numpy as np

class RidgeRegression:
    """
    Kernel Ridge Regression model.

    Attributes:
        ridge (float): Regularization parameter.
        N (int): Number of samples.
        Ky (numpy.ndarray): Coefficients of the fitted model.
        fit_mean (bool): Whether to fit the mean of the data.
        mean_y (float or numpy.ndarray): Mean of the target variable.
        mean_K (float or numpy.ndarray): Mean of the kernel matrix.
    """
    def __init__(self, ridge=0.0, fit_mean=False):
        """
        Initializes the KernelRidgeRegression model with specified parameters.

        Args:
            ridge (float, optional): Regularization parameter. Defaults to 0.0.
            fit_mean (bool, optional): Whether to fit the mean of the data. Defaults to False.
        """
        self.ridge = ridge
        self.N = None
        self.w = None
        self.fit_mean = fit_mean
    
    def fit(self, X, y):
        """
        Fits the model to the training data.

        Args:
            X (numpy.ndarray): Training feature design matrix.
            y (numpy.ndarray): Target variable.

        Notes:
            The method computes the coefficients of the model using the provided kernel matrix and target variable.
        """
        if self.fit_mean:
            self.mean_y = y.mean(0)
            self.mean_X = X.mean(0)
            X = X - self.mean_X[np.newaxis,:]
            y = y - self.mean_y
        else:
            self.mean_y = 0.0
        self.N = X.shape[0]
        XX = X.T @ X + np.eye(X.shape[1]) * self.ridge
        Xy = X.T @ y
        self.w = np.linalg.lstsq(XX, Xy)[0]
    
    def pred(self, X_star):
        """
        Predicts target variable for new data.

        Args:
            X_star (numpy.ndarray): Feature design matrix for new data.

        Returns:
            numpy.ndarray: Predicted target variable.
        """
        if self.fit_mean:
            X_star = X_star - self.mean_X[np.newaxis,:]
        return X_star @ self.w + self.mean_y
    
    def mse(self, X, y):
        """
        Computes mean squared error.

        Args:
            X (numpy.ndarray): Feature design matrix.
            y (numpy.ndarray): Target variable.

        Returns:
            float: Mean squared error.
        """
        y_pred = self.pred(X)
        residual = y - y_pred
        return np.mean(residual * residual)
    
    def score(self, X, y):
        """
        Computes the score of the model.

        Args:
            X (numpy.ndarray): Feature design matrix.
            y (numpy.ndarray): Target variable.

        Returns:
            float: Score of the model.
        """
        return self.mse(X=X, y=y)
