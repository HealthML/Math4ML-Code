import numpy as np
import matplotlib.pyplot as plt

def generate_dataset(w):
    x = np.linspace(-5,5, 15)
    y = w[0]*x + w[1] + np.random.normal(size=x.shape)

    x = np.append(x, [5, 4.5])
    y = np.append(y, [-4, -4])

    x = x[:,None]
    y = y[:,None]

    X = np.hstack((x, np.ones_like(x)))
    return X, y


def get_loss(X,y,w,p):
    pass

def get_gradient(X,y,w,p):
    pass

def get_gradient_finite_differences(X,y,w,p,e):
    pass


if __name__ == "__main__":
    # Generate dataset
    w_real = np.array([[1.75,1.2]]).T
    X, y = generate_dataset(w_real)

    # Compare the analytical gradient with the numerical gradient
    w_check = np.array([[1.7,1]]).T
    grad_analytical = get_gradient(X,y,w_check,2)
    grad_numerical = get_gradient_finite_differences(X,y,w_check,2,0.001)
    print(np.isclose(grad_analytical, grad_numerical))


    # Train the model using the gradient descent method
    w_solution=np.array([[1],[1]]) # starting solution
    p = 1
    lr = 0.001 # learning rate, feel free to adjust it, when the optimizer fails to converge
    loss = []
    e = 0.00001
    for i in range(10000):
        loss.append(get_loss(X,y,w_solution,p))
        grad = get_gradient(X,y,w_solution,p)
        w_solution = w_solution - lr*grad
        if len(loss)>1 and abs(loss[-1]-loss[-2])< e:
                break

    plt.figure(2)
    plt.plot(loss)

    plt.figure(3)
    plt.scatter(X[:,0],y)
    plt.plot(X[:,0], X@w_real)
    plt.plot(X[:,0],X@w_solution, c='r')
    plt.show()





