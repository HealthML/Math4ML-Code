import numpy as np
from scipy.spatial import distance_matrix

class Standardizer:
    """
    Standardizes the input data.

    Attributes:
        zero_mean (bool): Whether to center the data to have zero mean.
        unit_variance (bool): Whether to scale the data to have unit variance.
        mean (numpy.ndarray): Mean of each feature.
        standard_deviation (numpy.ndarray): Standard deviation of each feature.
    """
    def __init__(self, zero_mean=True, unit_variance=True):
        self.zero_mean = zero_mean
        self.unit_variance = unit_variance
        self.mean = 0
        self.standard_deviation = 1
    
    def fit(self, X):
        """
        Fits the standardizer to the input data.

        Args:
            X (numpy.ndarray): Input data.
        """
        if self.zero_mean:
            self.mean = X.mean(0)
        else:
            self.mean = np.zeros(X.shape[1])
        if self.unit_variance:
            self.standard_deviation = X.std(0)
        else:
            self.standard_deviation = np.ones(X.shape[1])
    
    def transform(self, X):
        """
        Transforms the input data using the fitted standardizer.

        Args:
            X (numpy.ndarray): Input data.

        Returns:
            numpy.ndarray: Transformed data.
        """
        return (X-self.mean[np.newaxis,:]) / self.standard_deviation[np.newaxis,:]
    
    def reverse_transform(self, X):
        """
        Reverse transforms the standardized data to the original scale.

        Args:
            X (numpy.ndarray): Standardized data.

        Returns:
            numpy.ndarray: Reverse transformed data.
        """
        return (X * self.standard_deviation[np.newaxis,:]) + self.mean[np.newaxis,:]


class MinkowskiExponentialKernel:
    """
    Minkowski Exponential Kernel function.

    Attributes:
        scale (float): Scaling factor for the kernel.
        length_scale (float): Length scale parameter.
        p (float): Exponent for the Minkowski distance.
        standardizer (Standardizer): Standardizer instance for preprocessing data.
        X (numpy.ndarray): Standardized input data.
    """
    def __init__(self, scale=1.0, length_scale=1.0, p=1.0, zero_mean=False, unit_variance=False):
        self.scale = scale
        self.length_scale = length_scale
        self.p = p
        self.standardizer = Standardizer(zero_mean=zero_mean, unit_variance=unit_variance)
    
    def fit(self, X):
        """
        Fits the kernel to the input data.

        Args:
            X (numpy.ndarray): Input data.
        """
        self.standardizer.fit(X)
        self.X = self.standardizer.transform(X)
        
    def transform(self, X_star):
        """
        Transforms new data using the kernel.

        Args:
            X_star (numpy.ndarray): New data to be transformed.

        Returns:
            numpy.ndarray: Transformed data.
        """
        X_star = self.standardizer.transform(X_star)
        distancematrix = distance_matrix(X_star, self.X, self.p)
        K = self.scale * np.exp(-np.power(distancematrix,self.p)/self.length_scale)
        return K

class SquaredExponentialKernel:
    """
    Squared Exponential Kernel function.

    Attributes:
        scale (float): Scaling factor for the kernel.
        length_scale (float): Length scale parameter.
        standardizer (Standardizer): Standardizer instance for preprocessing data.
        X (numpy.ndarray): Standardized input data.
        norm2_X (numpy.ndarray): Squared norms of the input data.
    """
    def __init__(self, scale=1.0, length_scale=1.0, zero_mean=False, unit_variance=False):
        self.scale = scale
        self.length_scale = length_scale
        self.standardizer = Standardizer(zero_mean=zero_mean, unit_variance=unit_variance)
    
    def fit(self, X):
        """
        Fits the kernel to the input data.

        Args:
            X (numpy.ndarray): Input data.
        """
        self.standardizer.fit(X)
        self.X = self.standardizer.transform(X)
        self.norm2_X = (X*X).sum(1)

    def transform(self, X_star):
        """
        Transforms new data using the kernel.

        Args:
            X_star (numpy.ndarray): New data to be transformed.

        Returns:
            numpy.ndarray: Transformed data.
        """
        X_star = self.standardizer.transform(X_star)
        XX = X_star @ self.X.T
        norm2_X_star = (X_star*X_star).sum(1)
        K = self.scale * np.exp((XX - 0.5 * self.norm2_X[np.newaxis,:] - 0.5 * norm2_X_star[:,np.newaxis])/self.length_scale)
        return K

class PolynomialKernel:
    """
    Polynomial Kernel function.

    Attributes:
        constant (float): Constant term in the polynomial.
        degree (float): Degree of the polynomial.
        standardizer (Standardizer): Standardizer instance for preprocessing data.
        X (numpy.ndarray): Standardized input data.
    """
    def __init__(self, constant=1.0, degree=1.0, zero_mean=False, unit_variance=False):
        self.constant = constant
        self.degree = degree
        self.standardizer = Standardizer(zero_mean=zero_mean, unit_variance=unit_variance)
    
    def fit(self, X):
        """
        Fits the kernel to the input data.

        Args:
            X (numpy.ndarray): Input data.
        """
        self.standardizer.fit(X=X)
        self.X = self.standardizer.transform(X=X)
        
    def transform(self, X_star):
        """
        Transforms new data using the kernel.

        Args:
            X_star (numpy.ndarray): New data to be transformed.

        Returns:
            numpy.ndarray: Transformed data.
        """
        X_star = self.standardizer.transform(X_star)
        XX = X_star @ self.X.T
        
        K = np.power(XX + self.constant, self.degree)
        if (self.degree % 1.0): # for non-integer degrees, we could have NaNs in the Kernel
            K[K!=K] = 0.0
        return K

class LinearKernel:
    """
    Linear Kernel function.

    Attributes:
        standardizer (Standardizer): Standardizer instance for preprocessing data.
        X (numpy.ndarray): Standardized input data.
    """
    def __init__(self, zero_mean=True, unit_variance=True):
        self.standardizer = Standardizer(zero_mean=zero_mean, unit_variance=unit_variance)
        self.X = None

    def fit(self, X):
        """
        Fits the kernel to the input data.

        Args:
            X (numpy.ndarray): Input data.
        """
        self.standardizer.fit(X=X)
        self.X = self.standardizer.transform(X=X)

    def transform(self,X_star):
        """
        Transforms new data using the kernel.

        Args:
            X_star (numpy.ndarray): New data to be transformed.

        Returns:
            numpy.ndarray: Transformed data.
        """
        X_star = self.standardizer.transform(X_star)
        return X_star @ self.X.T
