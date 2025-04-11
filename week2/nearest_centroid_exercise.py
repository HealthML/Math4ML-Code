# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "kagglehub",
#     "matplotlib",
#     "seaborn",
#     "numpy",
#     "pandas",
#     "scikit-learn",
# ]
# ///
import numpy as np


class NearestCentroidClassifier:
    def __init__(self):
        self.w = None  # the weight vector of the hyperplane
        self.b = None  # the bias vector of the hyperplane

    def fit(self, X, y):
        """
        Fit the model using X as training data and y as target values.
        Calculate the hyperplane splitting the vectorspace into the two classes.
        :param X: array-like, shape (n_samples, n_features) Training data.
        :param y: array-like, shape (n_samples,) Target values.
        """
        pass

    def predict(self, X):
        """
        Perform classification on samples in X using the hyperplane.
        :param X: array-like, shape (n_samples, n_features) Input data.
        :return: array, shape (n_samples,) Predicted class label per sample.
        """
        assert (self.w is not None) and (
            self.b is not None
        ), "You need to `.fit()` before being able to predict."
        pass

    def plot(self, X, y):
        """
        Perform classification on samples in X.
        Then plot both the decision boundary as a line and the samples as (x,y) scatter.
        Show the actual class as color. Show the predicted class as marker-shape.
        :param X: array-like, shape (n_samples, n_features) Input data.
        :param y: array-like, shape (n_samples,) True classes.
        :return: matplotlib.Figure, the figure of the described plot.
        """

        # Hint: You can use a dataframe to prepare the data to be plotted.
        # Hint: Use matplotlibs axline to plot a dashed line representing the decision boundary.
        #       Remember that you have to calculate this line first.
        # Hint: use seaborns scatterplot to plot the dataframe.
        import seaborn as sns
        import matplotlib.pyplot as plt
        import pandas as pd

        y_pred = self.predict(X)

        fig, ax = plt.subplots()
        # Your plotting code goes here :).

        return fig


def main():
    import pandas as pd
    import kagglehub
    from kagglehub import KaggleDatasetAdapter
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score

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
        ["concavity_mean", "texture_mean"]
    ]  # these two features work well acc. to Lit.
    # normalize the columns of x individually
    x = (x - x.min()) / (x.max() - x.min())

    # split data train 70 % and test 30 %
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.8, random_state=42
    )

    # Create and train the Nearest Centroid Classifier
    classifier = NearestCentroidClassifier()
    classifier.fit(x_train.to_numpy(), y_train.to_numpy())
    # Predict the classes for the test data
    y_pred = (classifier.predict(x_test.to_numpy()) + 1) / 2

    # Calculate and print the accuracy, then plot the result
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.3f}")
    classifier.plot(x_test.to_numpy(), y_test.to_numpy())
    plt.show()


# Example usage:
if __name__ == "__main__":
    main()
