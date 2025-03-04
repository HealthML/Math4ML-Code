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
    assert y.ndim==2, 'Wrong y dim'
    assert w.ndim==2, 'Wrong w dim'
    loss = np.sum(np.power(np.absolute(X@w-y),p)) / X.shape[0]
    return loss

def get_gradient(X,y,w,p):
    grad = p*(np.power(np.absolute((X@w-y)), p-1)*np.sign(X@w-y)).T@X /X.shape[0]
    return grad.T

def get_gradient_finite_differences(X,y,w,p,e):
    grad = np.zeros_like(w)
    for i in range(len(w)):
        w_diff = np.zeros_like(w)
        w_diff[i] += e/2
        grad[i] = (get_loss(X,y,w+w_diff,p) - get_loss(X,y,w-w_diff,p)) / e
    return grad


if __name__ == "__main__":
    
    np.random.seed(42)
    p = 1 


    # Generate dataset
    w_real = np.array([[1.75,1.2]]).T
    X, y = generate_dataset(w_real)

    # Compare the analytical gradient with the numerical gradient
    w_check = np.array([[1.7,1]]).T
    grad_analytical = get_gradient(X,y,w_check,p)
    grad_numerical = get_gradient_finite_differences(X,y,w_check,p,0.001)
    print(np.isclose(grad_analytical, grad_numerical))


    # Train the model with the gradient descent method
    w_solution=np.array([[1],[1]]) # starting solution
    lr = 0.01 # learning rate
    loss = []
    e = 0.000001
    for i in range(10000):
        loss.append(get_loss(X,y,w_solution,p))
        grad = get_gradient(X,y,w_solution,p)
        w_solution = w_solution - lr*grad
        if len(loss)>1 and abs(loss[-1]-loss[-2])< e:
                break
    plt.figure(2)
    plt.plot(loss)
    plt.title("Loss over iterations")

    
    plt.figure(3)
    plt.scatter(X[:,0],y)
    plt.plot(X[:,0], X@w_real, 'b-', label='Real parameters')
    plt.plot(X[:,0],X@w_solution, 'r-', label='Estimated parameters')
    plt.title("Real vs. Estimated Model")
    plt.xlabel("Input feature")
    plt.ylabel("Output")
    plt.legend()
    plt.show()





