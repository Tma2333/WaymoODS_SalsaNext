from builtins import range
from builtins import object
import numpy as np
from past.builtins import xrange

import collections

class KNearestNeighbor(object):
    """ a kNN classifier with L2 distance """

    def __init__(self):
        pass

    def train(self, X, y):
        """
        Train the classifier. For k-nearest neighbors this is just
        memorizing the training data.

        Inputs:
        - X: A numpy array of shape (num_train, D) containing the training data
          consisting of num_train samples each of dimension D.
        - y: A numpy array of shape (N,) containing the training labels, where
             y[i] is the label for X[i].
        """
        self.X_train = X
        self.y_train = y

    def predict(self, X, k):
        """
        Predict labels for test data using this classifier.

        Inputs:
        - X: A numpy array of shape (num_test, D) containing test data consisting
             of num_test samples each of dimension D.
        - k: The number of nearest neighbors that vote for the predicted labels.
        - num_loops: Determines which implementation to use to compute distances
          between training points and testing points.

        Returns:
        - y: A numpy array of shape (num_test,) containing predicted labels for the
          test data, where y[i] is the predicted label for the test point X[i].
        """

        dists = self.compute_distances_no_loops(X)

        return self.predict_labels(dists, k=k)
    

    def compute_distances_no_loops(self, X):
        """
        Compute the distance between each test point in X and each training point
        in self.X_train using no explicit loops.

        Input / Output: Same as compute_distances_two_loops
        """
        num_test = X.shape[0]
        num_train = self.X_train.shape[0]
        dists = np.zeros((num_test, num_train))

        # use (a - b)^2 = a^2 - 2(a*b) + b^2
        # So a is going to be the elements in test (X)
        # b the elements in train (self.X_train)

        term1 = np.sum(np.square(X), axis=1)
        term2 = -2*np.dot(X, self.X_train.T)
        term3 = np.sum(np.square(self.X_train), axis=1)
        combined = term1.reshape((term1.shape[0], 1)) + term2 + term3.reshape((1, term3.shape[0]))
        dists = np.sqrt(combined)

        return dists


    def predict_labels(self, dists, k):
        """
        Given a matrix of distances between test points and training points,
        predict a label for each test point.

        Inputs:
        - dists: A numpy array of shape (num_test, num_train) where dists[i, j]
          gives the distance betwen the ith test point and the jth training point.

        Returns:
        - y: A numpy array of shape (num_test,) containing predicted labels for the
          test data, where y[i] is the predicted label for the test point X[i].
        """

        def most_common(lst):
          # https://stackoverflow.com/questions/1518522/find-the-most-common-element-in-a-list
          data = collections.Counter(sorted(lst))  # The "sorted" asserts we fulfill the requirement
          # of picking out the smallest label in case of ties
          return max(lst, key=data.get)

        num_test = dists.shape[0]
        y_pred = np.zeros(num_test)
        for i in range(num_test):
            # A list of length k storing the labels of the k nearest neighbors to
            # the ith test point.
            closest_y = []
            
            inds = np.argsort(dists[i, :])  # Pick out indices of best train images after sorting the distances per image
            closest_y = self.y_train[inds[:k]]  # pick out labels of the k nearest neighbors

            y_pred[i] = most_common(closest_y)  # Pick out the most common label

        return y_pred
