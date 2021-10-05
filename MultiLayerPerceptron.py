# Rede Neural Artificial
import random as rand

import numpy

from DataProcessing import DataProcessing
import numpy as np
from sklearn.preprocessing import normalize


# h: numero de neuronios na camada escondida
# N - numero de Instâncias
# ne - numero de entradas de atributos
# ns - numero de saídas

def normalized(a, axis=-1, order=2):
    l2 = np.atleast_1d(np.linalg.norm(a, order, axis))
    l2[l2 == 0] = 1
    return a / np.expand_dims(l2, axis)


class MultiLayerPerceptron:

    def result(self, matrizDeDados):
        brutal = self.MLPfunction(matrizDeDados)
        for index, x in np.ndenumerate(brutal):
            if x > 0.5:
                brutal[index] = 1
            else:
                brutal[index] = 0

        return brutal

    def MLPfunction(self, matrizDeDados):
        N = len(matrizDeDados)
        Zin = matrizDeDados.__mul__(self.A.transpose())
        Z = np.matrix(1 / (1 + np.exp(Zin)))

        s = (N, 1)
        Z.__add__(np.ones(s, dtype=int))
        Yin = Z.__mul__(self.B.transpose())
        Yr = 1 / (1 + np.exp(Yin))
        return Yr

    def calcGrad(self, N, Ne, Ns):
        Zin = self.X.__mul__(self.A.transpose())
        Z = np.matrix(1 / (1 + np.exp(Zin)))

        s = (N, 1)
        Z.__add__(np.ones(s, dtype=int))
        Yin = Z.__mul__(self.B.transpose())
        Yr = 1 / (1 + np.exp(Yin))

        erro = Yr - self.Y
        gl = np.multiply(1 - Yr, Yr)

        dEdB = np.matrix(np.multiply(erro, gl)).transpose().__mul__(Z)

        dEdZ = np.matrix(np.multiply(erro, gl)).__mul__(self.B[:, 1:N - 1])

        dEdA = np.matrix(np.multiply(dEdZ, gl)).transpose().__mul__(self.X)

        return dEdA, dEdB

    def __init__(self, path, per, h):
        super().__init__()
        self.dp = DataProcessing(path, per)
        self.X = np.matrix(self.dp.Xtr)
        self.X.__add__([1, 1, 1, 1, 1, 1, 1])
        self.Y = np.matrix(self.dp.Ytr)
        self.Y.__add__([1])

        tam = np.size(self.X)
        N = len(self.X)
        Ne = int(tam / N)
        Ns = len(self.Y)
        self.A = np.random.rand(h, Ne)
        self.B = np.random.rand(Ns, h)

        Yr = self.MLPfunction(self.X)
        erro = (self.Y - Yr)
        E = np.sum(np.sum(np.multiply(erro, erro)))

        nepmax = 2000
        nep = 0
        alfa = 0.1

        [dEdA, dEdB] = self.calcGrad(N, Ne, Ns)
        dEdAnorm = normalized(dEdA)
        dEdBnorm = normalized(dEdA)
        g = [[dEdA[:], dEdB[:]]]

        while (np.any(dEdAnorm > 1e-3) or np.any(dEdBnorm > 1e-3)) and nep < nepmax and E > 1e-4:
            nep += 1
            self.A -= alfa * dEdA
            self.B -= alfa * dEdB

            [dEdA, dEdB] = self.calcGrad(N, Ne, Ns)
            g = np.array(dEdA[:] + dEdB[:])

            Yr = self.MLPfunction(self.X)
            erro = (self.Y - Yr)
            E = np.sum(np.sum(np.multiply(erro, erro)))
