import numpy as np
import matplotlib.pyplot as plt

def generate_dataset(w, noise=True, outliers=True):

    x = np.linspace(-5, 5, 15)
    y = w[0]*x + w[1]*x**2 + w[2] + (np.random.normal(size=x.shape) if noise else 0)

    # To make things spicier, we add some outliers
    if outliers:
        x = np.append(x, [4.8, -4.7])
        y = np.append(y, [15, -10])

    x = x[:, None]
    y = y[:, None]

    X = np.hstack((x, x**2, np.ones_like(x)))
    return X, y

def get_loss(X, y, w, p=2):
    """
    Compute loss.

    Parameters:
    X : np.ndarray
        Input data of shape (m, n).
    y : np.ndarray
        Target values of shape (m, 1).
    w : np.ndarray
        Weights of shape (n, 1).
    p : int
        Norm order (default is 2).

    Returns:
    loss : float
        Computed loss.

    """        
    
    pass


def get_gradient(X, y, w, p=2):

    """
    Compute gradient.

    Parameters:
    X : np.ndarray
        Input data of shape (m, n).
    y : np.ndarray
        Target values of shape (m, 1).
    w : np.ndarray
        Weights of shape (n, 1).
    p : int
        Norm order (default is 2).

    Returns:
    grad : np.ndarray
        Gradient of shape (n, 1).

    """  

    pass

def get_gradient_finite_differences(X, y, w, p=2, epsilon=1e-5):

    """
    Compute gradient using finite differences.

    Parameters:
    X : np.ndarray
        Input data of shape (m, n).
    y : np.ndarray
        Target values of shape (m, 1).
    w : np.ndarray
        Weights of shape (n, 1).
    p : int
        Norm order (default is 2).
    epsilon : float
        Small perturbation for finite differences.

    Returns:
    grad : np.ndarray
        Gradient of shape (n, 1).

    """

    pass

def compute_jacobian(f, x, epsilon=1e-5):
    
    """
    Compute Jacobian of vector-valued function f at point x using finite differences.
    
    Parameters:
    f : callable
        Function that takes a vector x and returns a vector y.
    x : np.ndarray
        Point at which to compute the Jacobian.
    epsilon : float
        Small perturbation for finite differences.

    Returns:
    J : np.ndarray
        Jacobian matrix of shape (m, n) where m is the size of f(x) and n is the size of x.

    
    """

    pass

if __name__ == "__main__":
    
    np.random.seed(42)
    p = 1 


    w_real = np.array([[1.0], [0.5], [2.0]])
    X, y = generate_dataset(w_real)

    # Check gradient
    w_check = np.array([[0.9], [0.4], [1.8]])
    grad_analytical = get_gradient(X, y, w_check)
    grad_numerical = get_gradient_finite_differences(X, y, w_check)
    grad_matched = np.allclose(grad_analytical, grad_numerical, atol=1e-4)
    assert grad_matched, "Gradient check failed!"
    print("Gradient check:", grad_matched)

    # Gradient descent
    w = np.random.randn(3,1)
    lr = 0.0001
    e = 0.000001
    losses = []
    for i in range(1000):
        losses.append(get_loss(X, y, w))
        grad = get_gradient(X, y, w)
        w -= lr * grad
        if i > 1 and abs(losses[-1] - losses[-2]) < e:
            break

    # Plot convergence
    plt.figure()
    plt.plot(losses)
    plt.title("Loss during training")

    # Predictions
    plt.figure()
    plt.scatter(X[:, 0], y, label="Data")
    x_vals = np.linspace(-5, 5, 100)[:, None]
    X_plot = np.hstack((x_vals, x_vals**2, np.ones_like(x_vals)))
    plt.plot(x_vals, X_plot @ w_real, label="True model")
    plt.plot(x_vals, X_plot @ w, 'r--', label="Learned model")
    plt.legend()
    plt.title("Model fit")
    plt.show()

    # Jacobian 
    def pred_func(w_flat):
        w_vec = w_flat.reshape(3,1)
        return X @ w_vec

    J = compute_jacobian(pred_func, w.flatten())
    print("Jacobian shape:", J.shape)
    print("Jacobian at w:", J)
