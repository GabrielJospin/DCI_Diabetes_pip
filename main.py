import numpy as np
from LinearRegression import LinearRegression
from LogicRegression import LogicRegression


def compare(original, resultado):
    error = 0
    for index, x in np.ndenumerate(original):
        if x != resultado[index[0]]:
            error += 1
    return error


Lr = LinearRegression("database/diabetes.csv", 0.7)
LogR = LogicRegression("database/diabetes.csv", 0.7)
dados = np.matrix(Lr.dp.Xts)

resultLogR = LogR.result(dados)
errosLogR = compare(np.matrix(Lr.dp.Yts), resultLogR)
resultLR = Lr.result(dados)
errosLR = compare(np.matrix(Lr.dp.Yts), resultLR)
total = len(Lr.dp.Yts)
print("Linear Regression: ")
print(1 - errosLR/total)
print("Logical Regression: ")
print(1 - errosLogR/total)

