
import numpy as np
import matplotlib.pyplot as plt

EPS = 0.001

def func(x1,x2):
    z = 2*np.cosh(x1) + np.cosh(x2) + np.cosh(0.1*x1*x2)
    return z


def func_grad(x1, x2):
    grad = # implement gradient
    return grad

def func_hessian(x1, x2):
    hessian = # implement Hessian
    return hessian

def draw_function():
  
    x = np.linspace(-2, 4.5, 100)
    y = np.linspace(-2, 4.5,  100)
    X, Y = np.meshgrid(x, y)
    Z = func(X,Y)
    plt.pcolormesh(X, Y, Z, cmap='viridis')
    plt.colorbar()
    plt.contour(X,Y,Z)
    plt.xlabel('x1')
    plt.ylabel('x2')

def gradient_descent():
    X = np.array([4,4])
    lr = 0.03
    solutions = np.copy(X)[:, np.newaxis]
    func_values = [func(X[0], X[1])]
    for i in range(500):

        X = # implement update

        solutions = np.concatenate((solutions, X[:,np.newaxis]), axis= 1)
        func_values.append(func(X[0], X[1]))
        if np.abs(func_values[-1]-func_values[-2]) < EPS:
            print(f'Gradient descent number of iterations: {i}')
            break

    plt.figure()
    draw_function()
    plt.plot(solutions[0,:], solutions[1,:])
    plt.scatter(solutions[0,:], solutions[1,:],s=4 )
    plt.title('Gradient descent')

def newtons_method():
    X = np.array([4,4])
    lr = 0.5
    solutions = np.copy(X)[:, np.newaxis]
    func_values = [func(X[0], X[1])]

    for i in range(500):

        X = # implement update

        solutions = np.concatenate((solutions, X[:,np.newaxis]), axis= 1)
        func_values.append(func(X[0], X[1]))
        if np.abs(func_values[-1]-func_values[-2]) < EPS:
            print(f'Newtons methods number of iterations: {i}')
            break
    plt.figure()
    draw_function()
    plt.plot(solutions[0,:], solutions[1,:])
    plt.scatter(solutions[0,:], solutions[1,:])

if __name__ == "__main__":

    gradient_descent()
    newtons_method()
    plt.show(block=True)






