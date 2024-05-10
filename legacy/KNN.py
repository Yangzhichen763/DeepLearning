import numpy as np


class NearestNeighbor:
    def __init__(self):
        pass

    def train(self, x, y):
        self.xtr = x
        self.ytr = y

    def predict(self, X):
        num_test = X.shape[0]
        y_prediction = np.zeros(num_test, dtype = self.ytr.dtype)

        for i in range(num_test):
            distances = np.sum(np.abs(self.xtr - X[i, :]), axis = 1)
            min_index = np.argmin(distances)
            y_prediction = self.ytr[min_index]

        return y_prediction

