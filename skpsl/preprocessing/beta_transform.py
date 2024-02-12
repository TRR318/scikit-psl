import numpy as np
from betacal import BetaCalibration
from sklearn.base import TransformerMixin
from sklearn.exceptions import NotFittedError
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler


class BetaTransformer(TransformerMixin):
    def __init__(self):
        self.clf = None

    def fit(self, X, y):
        self.clf = Pipeline([("scaler", MinMaxScaler()),
                        ("calibrator", BetaCalibration("abm"))]).fit(X.astype(float), y)
        return self

    def transform(self, X):
        if self.clf is None:
            raise NotFittedError()
        return self.clf.predict(X.astype(float))
