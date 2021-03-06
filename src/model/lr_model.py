from src.model.base import BasicLRModel
from joblib import dump, load
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np


class LRModel(BasicLRModel):
    def __init__(self, standardize=True):
        super().__init__()
        self.standardize = standardize
        self.scaler = StandardScaler()
        self.mapping_test = None

    def train(self, X, y):
        """
        Train a model with train data X and label y
        :param X: train data, shape = (Sample number, Feature number)
        :param y: train label, shape = (Sample number, 2): temperature of region 1 and temperature of region 2
        """
        super().train(X, y)
        self.model = LinearRegression()
        if self.standardize:
            X = self.scaler.fit_transform(X)
        self.model.fit(X, y)

    def predict(self, X_test: np.array or list) -> np.array:
        """
        Predict using trained model
        :param X_test: test data, shape = (Sample number, Feature number)
        :return: a array with exactly 2 number: temperature of region 1 and temperature of region 2
        """
        super().predict(X_test)
        X_test = np.array(X_test)
        if X_test.ndim == 1:
            X_test = X_test.reshape((1, len(X_test)))

        if self.standardize:
            X_test = self.scaler.transform(X_test)

        pred = self.model.predict(X_test)
        return pred

    def save(self, saved_path: str):
        super().save(saved_path)
        dump(self.model, saved_path + '.joblib')
        dump(self.scaler, saved_path + '.pkl')

    def load(self, loaded_path: str):
        super().load(loaded_path)
        self.model = load(loaded_path + '.joblib')
        self.scaler = load(loaded_path + '.pkl')
