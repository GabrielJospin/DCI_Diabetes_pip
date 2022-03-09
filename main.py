import numpy as np
from LinearRegression import LinearRegression
from LogicRegression import LogicRegression
from MultiLayerPerceptron import MultiLayerPerceptron


def compare(original, resultado):
    error = 0
    print(resultado)
    for index, x in np.ndenumerate(original):
        if x != resultado[index[0]]:
            error += 1
    return error


print("Carregando regressõa linear")
Lr = LinearRegression("database/diabetes.csv", 0.7)
print("Carregando Regresssão logica")
LogR = LogicRegression("database/diabetes.csv", 0.7)
print("Carregano MLP")
Mlp = MultiLayerPerceptron("database/diabetes.csv", 0.7, 2)

dados = np.matrix(Lr.dp.Xts)

resultLogR = LogR.result(dados)
errosLogR = compare(np.matrix(LogR.dp.Yts), resultLogR)
resultLR = Lr.result(dados)
errosLR = compare(np.matrix(Lr.dp.Yts), resultLR)
resultMLP = Mlp.result(dados)
errosMLP = compare(np.matrix(Mlp.dp.Yts), resultMLP)
total = len(Lr.dp.Yts)
print("Linear Regression: ")
print(1 - errosLR/total)
print("Logical Regression: ")
print(1 - errosLogR/total)
print("MLP: ")
print(1 - errosMLP/total)

