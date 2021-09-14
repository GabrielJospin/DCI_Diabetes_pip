import pandas as pd


class DBSM:

    def __init__(self, path):
        self.path = path
        self.readedFile = pd.read_csv(path)

    def update(self):
        self.readedFile.to_cvs(self.path)

