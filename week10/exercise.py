import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import os
os.environ["KERAS_BACKEND"] = "torch"
from keras.datasets import mnist




class PPCA():
    '''
    X - dataset
    x - data point
    z - laten variable
    
    '''
    def __init__(self,X, M):
        self.D = X.shape[1] # dimension of oryginal data points   
        self.M = M # dimension of reduced data point
        self.X = X #dataset
        self.calculate_parameters()
    def calculate_parameters(self):
        '''
        Determine parameteres of the model (mean, variance and W matrix). 
        Have to be overriden in child classes
        '''
        raise NotImplementedError 
    def sample_x(self):
        '''
        Sample from p(x) distribution
        '''
        mean = self.mean
        C = np.dot(self.W_ML, self.W_ML.T) + self.sigma * np.eye(self.D)
        distribution = stats.multivariate_normal(mean, C)
        return distribution.rvs()
    def sample_z(self):
        '''
        Sample from p(z) distribution
        '''
        model = stats.multivariate_normal(np.zeros(shape = self.M), np.eye(self.M))
        return distribution.rvs()
    def sample_x_given_z(self, z):
        '''
        Sample from p(x|z) distribution'
        '''
        distribution = stats.multivariate_normal(np.dot(self.W_ML, z) + self.mean, self.sigma * np.eye(self.D))
        return distribution.rvs()
    def sample_z_given_x(self, x):
        '''
        Sample from p(z|x) distribution
        '''
        M_matrix = np.dot(self.W_ML.T, self.W_ML) + self.sigma * np.eye(self.M)
        M_matrix_inv = np.linalg.inv(M_matrix)
        mean = np.linalg.multi_dot([M_matrix_inv, self.W_ML.T, (x - self.mean)])
        variance = self.sigma * M_matrix_inv                                    
        distribution = stats.multivariate_normal(mean, variance)
        return distribution.rvs()
                                       

if __name__ == "__main__":
    # Sample from p(x) distribution

    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train / 255
    x_train = x_train.reshape(60000, -1)
    x_train = x_train[((y_train == 8) + (y_train == 1)),:]
    y_train = y_train[((y_train == 8) + (y_train == 1))]

    model = PPCA(x_train, 2)

    # sample from p(x)
    plt.figure(figsize =(10,10))
    for i in range(1,10):
        plt.subplot(3,3,i)
        plt.imshow(model.sample_x().reshape(28,28))
    plt.suptitle('Sampling from p(x)', fontsize=20)

    # Show the original image
    plt.figure()
    idx = np.random.randint(0, x_train[0].shape[0])
    plt.imshow(x_train[idx,:].reshape(28,28))
    plt.suptitle('Original image', fontsize=20)

    # Show the reconstructions

    z = model.sample_z_given_x(x_train[idx,:]) # get latent variable p(z|x)

    plt.figure(figsize =(10,10))
    for i in range(1,10):
        plt.subplot(3,3,i)
        image = model.sample_x_given_z(z)
        plt.imshow(image.reshape(28,28))
        plt.axis('off')
    plt.tight_layout()
    plt.suptitle('Image reconstruction p(x|z)', fontsize=20)
    plt.show()
