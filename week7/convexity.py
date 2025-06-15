import numpy as np
import matplotlib.pyplot as plt

ml_functions ={
    "MSE": lambda x: (x - 3)**2,
    "Logistic Loss": lambda x: np.log(1 + np.exp(-x)),
    "Hinge Loss": lambda x: np.maximum(0, 1 - x),
    "L2 Regularization": lambda x: x**2,
    "Exponential Loss": lambda x: np.exp(-x),

    "Sinusoidal": lambda x: np.sin(x),
    "Non-Convex Combo": lambda x: np.sin(x) + 0.1*x**2,
    "ReLU": lambda x: np.maximum(0, x)
}


def plot_with_secant(f, name):
    x_vals = np.linspace(-5, 5, 500)
    y_vals = f(x_vals)

    # Random convex combination
    x1, x2 = np.random.uniform(-4, 4, 2)
    x1, x2 = sorted([x1, x2])
    alpha = np.random.uniform(0, 1)
    x_alpha = alpha * x1 + (1 - alpha) * x2

    y1, y2 = f(x1), f(x2)
    y_alpha = f(x_alpha)
    secant_val = alpha * y1 + (1 - alpha) * y2

    plt.plot(x_vals, y_vals, label=f"{name}")
    plt.plot([x1, x2], [y1, y2], 'r--', label='Secant line')
    plt.scatter([x_alpha], [y_alpha], color='blue', label='f(αx1 + (1-α)x2)')
    plt.scatter([x_alpha], [secant_val], color='green', label='αf(x1)+(1-α)f(x2)')
    plt.title(f"{name}: Visual Jensen's Inequality")
    plt.legend()
    plt.grid(True)


def second_derivative(f, x, h=1e-4):
    try:
        return (f(x + 2*h) - 2 * f(x+h) + f(x)) / (h ** 2)
    except:
        return None

def is_convex(f, x_range=(-5, 5), num_points=1000):
    x_vals = np.linspace(x_range[0], x_range[1], num_points)
    for x in x_vals:
        pass
        # your code here

    # return True or False based on convexity

def is_convex_jensen(f, x_range=(-5, 5), num_samples=1000):
    for i in range(num_samples):
        pass
        # your code here
        # Hint: generate two uniform random points (x1, x2) using x_range
            # sort x1, x2 points
            # sample alpha uniformly from [0, 1]
            #  apply Jensen's inequality
        # You can use tol for numerical stability in comparisons (oiptional)

    # return True or False based on Jensen's inequality

if __name__ == "__main__":
    print("Convexity Check (2nd Derivative Test)\n")
    for name, f in ml_functions.items():
        convex_2nd = is_convex(f)
        convex_jensen = is_convex_jensen(f)
        print(f"{name:<15}: 2nd Derivative Test: {'✅' if convex_2nd else '❌'}, Jensen's Test: {'✅' if convex_jensen else '❌'}")

    print("\n🔍 Visual Convexity Checks (Secant Line Test)\n")
    plt.figure(figsize=(8, 6))
    for idx, (name, f) in enumerate(ml_functions.items()):
        plt.subplot(3, 3, idx + 1)
        plt.xlabel("x")
        plot_with_secant(f, name)
    plt.tight_layout()
    plt.show()
