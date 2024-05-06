import numpy as np

class KernelRidgeRegressionEx:
    """
    Kernel Ridge Regression model.

    Attributes:
        ridge (float): Regularization parameter.
        N (int): Number of samples.
        Ky (numpy.ndarray): Coefficients of the fitted model.
    """
    def __init__(self, ridge=0.0):
        """
        Initializes the KernelRidgeRegression model with specified parameters.

        Args:
            ridge (float, optional): Regularization parameter. Defaults to 0.0.
        """
        self.ridge = ridge
        self.N = None
        self.Ky = None
    
    def fit(self, K, y):
        """
        Fits the model to the training data.

        Args:
            K (numpy.ndarray): Kernel matrix.
            y (numpy.ndarray): Target variable.

        Notes:
            The method computes the coefficients of the model using the provided kernel matrix and target variable.
        """
        self.N = K.shape[0]
        # self.Ky = Please implement me as an exercise
    
    def pred(self, K_star):
        """
        Predicts target variable for new data.

        Args:
            K_star (numpy.ndarray): Kernel matrix for new data.

        Returns:
            numpy.ndarray: Predicted target variable.
        """
        # prediction = Please implement me as an exercise
        return prediction
    
    def mse(self, K, y):
        """
        Computes mean squared error.

        Args:
            K (numpy.ndarray): Kernel matrix.
            y (numpy.ndarray): Target variable.

        Returns:
            float: Mean squared error.
        """
        y_pred = self.pred(K)
        residual = y - y_pred
        return np.mean(residual * residual)
    
    def score(self, K, y):
        """
        Computes the score of the model.

        Args:
            K (numpy.ndarray): Kernel matrix.
            y (numpy.ndarray): Target variable.

        Returns:
            float: Score of the model.
        """
        return self.mse(K=K, y=y)
