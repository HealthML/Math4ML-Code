#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "kagglehub",
#     "matplotlib",
#     "numpy",
#     "pandas",
#     "scikit-learn",
# ]
# ///
import numpy as np
from sklearn.metrics import roc_auc_score


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def logistic_loss(yh, y):
    yh = np.clip(yh, 1e-10, 1 - 1e-10)  # prevents log(0)
    return -np.mean(y * np.log(yh) + (1 - y) * np.log(1 - yh))


def predict(X, w):
    return sigmoid(X @ w)


def newton_logistic_regression(data, w, max_iters=1000, tol=1e-8):
    X, y = data
    n_samples, n_features = X.shape
    w = np.zeros(n_features)  # initialize weights

    # for a maximum of `max_iter` do
        # predict with the current weights
        yh =
        # Calculate the gradient.
        gradient = 
        # Calculate the Hessian.
        hessian = 
        # Solve the system.
        delta = 
        # update the weights
        w = 
        # check for convergence and break if |delta|<tol

    return w


def apply_to_generated():
    from sklearn.datasets import make_classification
    from sklearn.preprocessing import StandardScaler

    # Generate synthetic binary classification data
    X, y = make_classification(
        n_samples=200, n_features=2, n_informative=2, n_redundant=0, random_state=42
    )

    # Add bias term
    X = np.hstack([np.ones((X.shape[0], 1)), X])  # Add intercept (bias) term

    # Normalize for better numerical behavior
    X = StandardScaler().fit_transform(X)

    np.random.seed(42)
    print("=" * 5, "2nd-order LR | Generated Data", "=" * 5)
    weights = np.random.randn(3)
    yh = predict(X, weights)
    print("-" * 10, "Before Fitting", "-" * 10)
    print(f"Loss: {logistic_loss(yh,y):.3f}")
    print(f"AUROC: {roc_auc_score(y,yh):.3f}")
    # Train
    weights = newton_logistic_regression((X, y), weights)
    yh = predict(X, weights)
    print("-" * 10, "After Fitting", "-" * 10)
    print(f"Loss: {logistic_loss(yh,y):.3f}")
    print(f"AUROC: {roc_auc_score(y,yh):.3f}")
    print("Learned weights:", weights)
    print()


def apply_to_breast_cancer():
    import pandas as pd
    import kagglehub
    from kagglehub import KaggleDatasetAdapter
    from sklearn.model_selection import train_test_split

    # Load a DataFrame with a specific version of a CSV
    data = kagglehub.dataset_load(
        KaggleDatasetAdapter.PANDAS,
        "uciml/breast-cancer-wisconsin-data/versions/2",
        "data.csv",
    )

    # y includes our labels and x includes our features
    y = data.diagnosis  # M or B
    y = y == "M"
    x = data[
        [
            "concavity_mean",
            "texture_mean",
            "radius_mean",
            "perimeter_mean",
            "area_mean",
            "smoothness_mean",
            "concavity_mean",
        ]
    ]  # these two features work well acc. to Lit.
    # normalize the columns of x individually
    x = (x - x.min()) / (x.max() - x.min())

    # split data train 70 % and test 30 %
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.8, random_state=42
    )
    np.random.seed(42)
    print("=" * 5, "2nd-order LR | Breast Cancer Data", "=" * 5)
    weights = np.random.randn(7)
    yh = predict(x_train, weights)
    print("-" * 10, "Before Fitting", "-" * 10)
    print(f"Loss (train): {logistic_loss(yh,y_train):.3f}")
    print(f"AUROC (train): {roc_auc_score(y_train,yh):.3f}")
    # Train
    weights = newton_logistic_regression(
        (x_train.to_numpy(), y_train.to_numpy()), weights
    )
    yh = predict(x_train, weights)
    print("-" * 10, "After Fitting", "-" * 10)
    print(f"Loss (train): {logistic_loss(yh,y_train):.3f}")
    print(f"AUROC (train): {roc_auc_score(y_train,yh):.3f}")
    yh_test = predict(x_test, weights)
    print(f"Loss (test): {logistic_loss(yh_test,y_test):.3f}")
    print(f"AUROC (test): {roc_auc_score(y_test,yh_test):.3f}")
    print("Learned weights:", weights)
    print()


def main():
    errors = []
    try:
        apply_to_generated()
    except Exception as e:
        errors.append(e)

    try:
        apply_to_breast_cancer()
    except Exception as e:
        errors.append(e)

    if not errors:
        return 0
    else:
        for e in errors:
            print(e)
        return 1


if __name__ == "__main__":
    main()
