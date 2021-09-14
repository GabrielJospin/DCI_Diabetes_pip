from random import randrange

import numpy
import numpy as np
import pandas as pd

import DBSM as db


class DataProcessing:

    def __init__(self, path, per):
        self.db = db.DBSM(path)
        self.X = self.db.readedFile.iloc[:, 1:8]
        self.Y = self.db.readedFile.iloc[:, 8:9]
        self.per = per
        n = len(self.X)

        Epos = self.db.readedFile.query('Outcome==1')
        Eneg = self.db.readedFile.query('Outcome==0')
        Xpos = Epos.iloc[:, 1:8]
        Ypos = Epos.iloc[:, 8:9]
        Xneg = Eneg.iloc[:, 1:8]
        Yneg = Eneg.iloc[:, 8:9]
        nPos = len(Xpos)
        nNeg = len(Xneg)
        nPosTr = int(nPos * self.per)
        nNegTr = int(nNeg * self.per)
        nTr = nPosTr + nNegTr
        nTs = n - nTr
        view = pd.array([1, 1, 1, 1, 1, 1, 1, 1])
        Yview = pd.array([1])

        self.Xtr = Xpos[1:nPosTr].combine_first(Xneg[1:nPosTr])
        self.Xts = Xpos[nPosTr + 1:].combine_first(Xneg[nPosTr + 1:])
        self.Ytr = Ypos[1:nPosTr].combine_first(Yneg[1:nPosTr])
        self.Yts = Ypos[nPosTr + 1:].combine_first(Yneg[nPosTr + 1:])
