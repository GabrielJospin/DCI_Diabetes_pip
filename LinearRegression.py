from DataProcessing import DataProcessing
import numpy as np


class LinearRegression:

    def __init__(self, path, per):
        self.dp = DataProcessing(path, per)
        self.X = np.matrix(self.dp.Xtr)
        self.X.__add__([1, 1, 1, 1, 1, 1, 1])
        self.Y = np.matrix(self.dp.Ytr)
        self.Y.__add__([1])

        Xt = self.X.transpose()
        Xc = np.matmul(np.linalg.inv(np.matmul(Xt, self.X)), Xt)
        self.w = np.matmul(Xc, self.Y)

    def linearFun(self, matrizDados):
        return np.matmul(matrizDados, self.w)

    def result(self, matrizDados):
        brutal = self.linearFun(matrizDados)
        for index, x in np.ndenumerate(brutal):
            if x >= 0.5:
                brutal[index] = 1
            else:
                brutal[index] = 0

        return brutal



