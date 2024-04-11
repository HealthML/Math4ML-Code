import numpy as np

class NearestCentroidClassifier:
    def __init__(self):
        
    
    def fit(self, X, y):
        """
        Fit the model using X as training data and y as target values.
        :param X: array-like, shape (n_samples, n_features) Training data.
        :param y: array-like, shape (n_samples,) Target values.
        """
        
    
    def predict(self, X):
        """
        Perform classification on samples in X.
        :param X: array-like, shape (n_samples, n_features) Input data.
        :return: array, shape (n_samples,) Predicted class label per sample.
        """
        


# Example usage:
if __name__ == "__main__":

    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score
    data = pd.read_csv('./../datasets/breast_cancer_data/data_processed.csv')
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