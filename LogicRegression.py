from math import exp, sqrt

from DataProcessing import DataProcessing
import numpy as np


class LogicRegression:

    @staticmethod
    def normalize(X: np.array):
        tam = len(X)
        Y = (X - X.mean()) / X.std()
        return Y

    def __normalize__(self):
        return (self.X - self.X.mean()) / self.X.std()

    def predict(self, X):
        X = self.normalize(X)
        result = []
        for (i, j), x in np.ndenumerate(X):
            if j == 0:
                poli = self.b1.__matmul__(np.matrix(X[i]))
                w = -self.b0 - np.sum(poli)
                factor = 1 / (1 + exp(w))
                result.append(factor)
        return result

    def result(self, matrizDados):
        brutal = self.predict(matrizDados)
        for index, ele in np.ndenumerate(np.array(brutal)):
            if ele >= 0.5:
                brutal[index[0]] = 1
            else:
                brutal[index[0]] = 0

        return brutal

    def __init__(self, path, per):
        self.dp = DataProcessing(path, per)
        self.X = np.matrix(self.dp.Xtr)
        self.X = self.__normalize__()
        self.X.__add__([1, 1, 1, 1, 1, 1, 1])
        self.Y = np.matrix(self.dp.Ytr)
        self.Y.__add__([1])
        self.b0 = 0
        self.b1 = np.matrix([0, 0, 0, 0, 0, 0, 0]).transpose()
        L = 0.001
        epochs = len(self.X)
        D_b0 = 0
        D_b1 = 0

        for epoch in range(epochs):
            y_pred = np.matrix(self.predict(self.X)).transpose()
            temp_Y = np.array(self.Y)
            temp_X = np.array(self.X)
            D_b0 = -2 * sum((temp_Y - y_pred) * y_pred.transpose() * (1 - y_pred))
            D_b1 = -2 * ((temp_X.transpose() * (temp_Y - y_pred)) * (y_pred.transpose() * (1 - y_pred)))

            self.b0 = self.b0 - L * D_b0[0, 0]
            self.b1 = np.matrix(self.b1 - L * D_b1)
