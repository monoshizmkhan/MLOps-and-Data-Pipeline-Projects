import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np
import os

class dataLoader:
    def __init__(self, portion=0.7):
        self.df = None
        self.portion = portion
        self.train = None
        self.test = None
        self.val = None
        self.scaler = StandardScaler()

    def loadRawData(self):
        base_path = os.path.dirname(os.path.abspath(__file__))  # folder of dataLoader.py
        file_path = os.path.join(base_path, "Boston.csv")
        self.df = pd.read_csv(file_path)

    def getData(self):
        return self.df

    def splitData(self):
        self.train = self.df.sample(frac=self.portion)
        df = self.df.drop(self.train.index)
        self.val = df.sample(frac=0.5)
        self.test = df.drop(self.val.index)

    def getSplits(self):
        return self.train, self.val, self.test

    def processData(self, df, train_scaler=False):
        y = df["MEDV"].tolist()
        x = df.drop("MEDV", axis=1)
        x = np.array(x)
        y = np.array(y)
        if train_scaler:
            self.scaler.fit(x)
        x = self.scaler.transform(x)
        return x, y

    def pipeline(self):
        self.loadRawData()
        self.splitData()
        train_x, train_y = self.processData(self.train, train_scaler=True)
        val_x, val_y = self.processData(self.val)
        test_x, test_y = self.processData(self.test)
        return train_x, train_y, val_x, val_y, test_x, test_y