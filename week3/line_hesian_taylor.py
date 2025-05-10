import numpy as np

def f(x):
    return x[0]**2 + 3 * x[0] * x[1] + 2 * x[1]**2

def grad_f(x):
    return np.array([2*x[0] + 3*x[1], 3*x[0] + 4*x[1]])

# ---- Hessian Estimation via Finite Differences ----
def hessian(f, x, epsilon=1e-5):
    n = len(x)
    H = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            x_ijp = x.copy()
            x_ijp[i] += epsilon
            x_ijp[j] += epsilon
            x_ipj = x.copy()
            x_ipj[i] += epsilon
            x_ipj[j] -= epsilon
            x_imj = x.copy()
            x_imj[i] -= epsilon
            x_imj[j] -= epsilon
            x_ijm = x.copy()
            x_ijm[i] -= epsilon
            x_ijm[j] += epsilon
            # TODO: Add code here to compute second-order partial derivative
            H[i, j] = ( )
    return H

# ---- Backtracking Line Search ----
def backtracking_line_search(x, grad, f, alpha=1.0, rho=0.5, c=1e-4):
    while True:
        # TODO: Add Armijo condition here to break the loop when satisfied
        if ():
            break
        alpha *= rho
    return alpha

# ---- Second-order Taylor Approximation ----
def taylor_approx(fx, grad, hess, delta):
    # TODO: Add code here to implement second-order Taylor expansion
    return ()

if __name__ == "__main__":
    x0 = np.array([1.0, 1.0])
    grad = grad_f(x0)
    H = hessian(f, x0)
    alpha = backtracking_line_search(x0, grad, f)
    delta = -alpha * grad          # Gradient descent step

    x1 = x0 + delta # New point
    actual_f = f(x1)
    approx_f = taylor_approx(f(x0), grad, H, delta)

    print("Initial x:", x0)
    print("Step size (alpha):", alpha)
    print("Next x:", x1)
    print("Actual f(x1):", actual_f)
    print("Taylor Approximation of f(x1):", approx_f)


    print("\n🔍 Running Tests:")
    expected_H = np.array([[2.0, 3.0], [3.0, 4.0]])
    assert np.allclose(H, expected_H, atol=1e-2)
    print("✅ Hessian test passed.")
    assert 0 < alpha <= 1.0
    print("✅ Line search step size test passed.")
    assert np.abs(actual_f - approx_f) < 1e-2
    print("✅ Taylor approximation test passed.")
