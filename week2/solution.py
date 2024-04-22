import numpy as np
from numpy.linalg import norm    
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


class NearestCentroidClassifier:
    def __init__(self):
        self.centroids = None
        self.classes_ = None
        
    
    def fit(self, X, y):
        """
        Fit the model using X as training data and y as target values.
        :param X: array-like, shape (n_samples, n_features) Training data.
        :param y: array-like, shape (n_samples,) Target values.
        """
        self.classes_ = np.unique(y)
        self.centroids = [ np.mean(X[y==class_], axis=0) for class_ in self.classes_]

        
    
    def predict(self, X):
        """
        Perform classification on samples in X.
        :param X: array-like, shape (n_samples, n_features) Input data.
        :return: array, shape (n_samples,) Predicted class label per sample.
        """
        differences_cent_0 = norm(X - self.centroids[0], axis=1)
        differences_cent_1 = norm(X - self.centroids[1], axis=1)
        output = np.where(differences_cent_1<differences_cent_0, self.classes_[1], self.classes_[0])
        return output


# Example usage:
if __name__ == "__main__":


    data = pd.read_csv('./datasets/breast_cancer_data/data_processed.csv')
    print(data.shape)
    # y includes our labels and x includes our features
    y = data.diagnosis      # M or B 
    list = ['diagnosis']
    x = data.drop(list,axis = 1 )

    # split data train 70 % and test 30 %
    x["bias"] = np.ones(x.shape[0])
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.8, random_state=42)

    #random forest classifier with n_estimators=10 (default)


    # Create and train the Nearest Centroid Classifier
    classifier = NearestCentroidClassifier()
    classifier.fit(x_train.values,y_train.values)
    # Predict the classes for the test data
    y_pred = classifier.predict(x_test.values)

    # Calculate and print the accuracy
    print("Accuracy:", accuracy_score(y_test, y_pred))