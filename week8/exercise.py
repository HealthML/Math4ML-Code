import numpy as np
import time
import sklearn.datasets
from tqdm import tqdm


def sigmoid(a):
    """
    returns the logistic sigmoid \pi(a)
    Keyword arguments:
    a -- scalar or numpy array
    """
    expa = np.exp(a)
    res = expa / (1.0 + expa)
    if hasattr(a, "__iter__"):
        res[a>709.7] = 1.0 # np.exp will overflow and return inf for values larger 709.7.
    elif a>709.7:
        res = 1.0
    return res

class LogisticRegression():
    def __init__(self, l2=0.01, num_iter = 100, method='gd', lr=0.001, tol=0.001) -> None:
        self.w = None
        self.num_iter = num_iter
        self.method = method
        self.lr = lr
        self.tol = tol
        self.l2 = l2


    def fit(self, X, y):
        self.class_labels = np.unique(y)
        self.w = np.zeros((X.shape[1], 1)) 
        if len(self.class_labels)>2:
            raise Exception("too many classes. This logistic regression class only implements binary classification.")

        objective_values = [self.objective(X,y)]

        for i in range(self.num_iter):

            gradient = self.perform_update(X, y)

            objective = self.objective(X,y)
            objective_values.append(objective)

            if np.max(np.abs(gradient)) < self.tol:
                print(f'Method: {self.method} Number of iterations: {i}')
                print(f'Objective function value: {objective_values[-1]}')
                break
        else:
            print(f'Maximum number of iterations reached, objective function value {objective_values[-1]}')
        self.training_length = i
        
    
    def predict_proba(self, X):
        return sigmoid(np.dot(X, self.w))

    def predict_proba_w(self, X, w):
        return sigmoid(np.dot(X, w))

    def perform_update(self, X, y):
        pi = self.predict_proba(X)



        if self.method == 'backtracking':
            #perform gradient descent update

            t=1
            alpha = 0.1
            beta = 0.3

            gradient = self.gradient(X,y, pi)
            i = 0
            while True: 
                # equation from the lecture https://github.com/HealthML/Math4ML-Lecture/blob/master/math4ml_2_Calculus_05_Unconstrained_Optimization_Convexity_handout.pdf
                left_side = self.objective_w(X,y,self.w-t*gradient)
                right_side = self.objective_w(X,y,self.w) - alpha*t*np.dot(gradient.T, gradient)
                if (left_side < right_side) or t<0.001:
                    if t<0.01:
                        print('Small t reached')
                    break
                t = t*beta
                i +=1

            update = - gradient * t


        if self.method == 'gd':
            #perform gradient descent update
            gradient = self.gradient(X,y, pi)
            update = - gradient * self.lr
        
        if self.method=='hessian':
            # implement full Hessian method of optimization here and save it in the 'update' variable
            

        if self.method=='diagonal_hessian':
            # implement diagonal Hessian method of optimization here and save it in the 'update' variable
            

        if self.method=='efficient_diagonal_hessian':
            # can you think of a more efficient way of implementing the diagonal Hessian method?
            

        self.w = self.w +  update
        return gradient


    
    def objective(self, X, y):
        pi = self.predict_proba(X)

        eps = np.finfo(pi.dtype).eps
        pi = np.clip(pi, eps, 1-eps) # to avoid (log(0))

        log_0_pi = np.log(pi[y==self.class_labels[1]])
        log_1_pi = np.log(1.0-pi[y==self.class_labels[0]])
        loss = -log_0_pi.mean() - log_1_pi.mean() # this version is more stable for perfect prediction

        regularizer = 0.5 * (self.l2 * self.w * self.w).sum()

        return loss + regularizer
        
    def objective_w(self, X, y,w):
        pi = self.predict_proba_w(X,w)

        eps = np.finfo(pi.dtype).eps
        pi = np.clip(pi, eps, 1-eps) # to avoid (log(0))

        log_0_pi = np.log(pi[y==self.class_labels[1]])
        log_1_pi = np.log(1.0-pi[y==self.class_labels[0]])
        loss = -log_0_pi.mean() - log_1_pi.mean() # this version is more stable for perfect prediction

        regularizer = 0.5 * (self.l2 * w * w).sum()

        return loss + regularizer
    
    def gradient(self, X, y, pi):
        gradient = np.dot(X.T, pi - (y==self.class_labels[1])[:,None] )/X.shape[0] +  self.l2 * self.w
        return gradient

    def hessian(self, X, y, pi):
        # write a formula for full Hessian here
        hessian = 
        return hessian 

    def hessian_diag(self, X, y, pi):
        # write a formula for diagonal Hessian here
        hessian_diag = 
        return hessian_diag

    




if __name__ == "__main__":
    repeat = 10
    n_samples = 1000
    n_features = 400
    n_informative = 400
    tol = 0.01

    t_start = time.time()
    np.random.seed(10)
    iterations = []
    for i in range(repeat):
        X, y = sklearn.datasets.make_classification(n_samples=n_samples, n_classes=2, n_features=n_features, n_informative=n_informative, n_redundant=0, class_sep=1, n_clusters_per_class=2)
        log = LogisticRegression(l2=0.5, lr=1, num_iter=1000, method='backtracking', tol=tol  )
        log.fit(X,y)
        iterations.append(log.training_length)
    t_end = time.time()
    print(f'Average time: {(t_end- t_start)/repeat}, average iterations {np.mean(iterations)}')

    print('\n')
    t_start = time.time()
    np.random.seed(10)
    iterations = []
    for i in range(repeat):
        X, y = sklearn.datasets.make_classification(n_samples=n_samples, n_classes=2, n_features=n_features, n_informative=n_informative, n_redundant=0, class_sep=1, n_clusters_per_class=3)
        log = LogisticRegression(l2=0.5, lr=1, num_iter=1000, method='hessian', tol=tol  )
        log.fit(X,y)
        iterations.append(log.training_length)
    t_end = time.time()
    print(f'Average time: {(t_end- t_start)/repeat}, average iterations {np.mean(iterations)}')

    print('\n')

    t_start = time.time()
    np.random.seed(10)
    iterations = []
    for i in range(repeat):
        X, y = sklearn.datasets.make_classification(n_samples=n_samples, n_classes=2, n_features=n_features, n_informative=n_informative, n_redundant=0, class_sep=2, n_clusters_per_class=2)
        log = LogisticRegression(l2=0.1, lr=0.2, num_iter=10000, method='gd', tol=tol)
        log.fit(X,y)
        iterations.append(log.training_length)
    t_end = time.time()
    print(f'Average time: {(t_end- t_start)/repeat}, average iterations {np.mean(iterations)}')

    print('\n')
    t_start = time.time()
    np.random.seed(10)
    iterations = []
    for i in range(repeat):
        X, y = sklearn.datasets.make_classification(n_samples=n_samples, n_classes=2, n_features=n_features, n_informative=n_informative, n_redundant=0, class_sep=2, n_clusters_per_class=2)
        log = LogisticRegression(l2=0.1, lr=0.25 ,num_iter=1001, method='diagonal_hessian', tol=tol)
        log.fit(X,y)
        iterations.append(log.training_length)
    t_end = time.time()
    print(f'Average time: {(t_end- t_start)/repeat}, average iterations {np.mean(iterations)}')

    print('\n')
    t_start = time.time()
    np.random.seed(10)
    iterations = []
    for i in range(repeat):
        X, y = sklearn.datasets.make_classification(n_samples=n_samples, n_classes=2, n_features=n_features, n_informative=n_informative, n_redundant=0, class_sep=2, n_clusters_per_class=2)
        log = LogisticRegression(l2=0.1, lr=0.25 ,num_iter=1000, method='efficient_diagonal_hessian', tol=tol)
        log.fit(X,y)
        iterations.append(log.training_length)
    t_end = time.time()
    print(f'Average time: {(t_end- t_start)/repeat}, average iterations {np.mean(iterations)}')