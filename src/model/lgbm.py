from src.model.base import BasicModel
from joblib import dump, load
import numpy as np


class LGBModel(BasicModel):
    def __init__(self):
        super().__init__()
        self.model = None

    def train(self, X, y):
        """
        Train a model with train data X and label y
        :param X: train data, shape = (Sample number, Feature number)
        :param y: train label, shape = (Sample number, 2): temperature of region 1 and temperature of region 2
        """
        # TODO: 暂时不提供训练接口
        pass

    def predict(self, X_test: np.array or list) -> np.array:
        """
        Predict using trained model
        :param X_test: test data, shape = (Sample number, Feature number)
        :return: a array with exactly 2 number: temperature of region 1 and temperature of region 2
        """
        X_test = np.array(X_test)
        if X_test.ndim == 1:
            X_test = X_test.reshape((1, len(X_test)))
        return self.model.predict(X_test)

    def save(self, saved_path: str):
        super().save(saved_path)
        dump(self.model, saved_path + '.joblib')

    def load(self, loaded_path: str):
        super().load(loaded_path)
        self.model = load(loaded_path + '.joblib')
