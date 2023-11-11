from sklearn.base import TransformerMixin
from sklearn.exceptions import NotFittedError
from sklearn.linear_model import LogisticRegression


class SigmoidTransformer(TransformerMixin):
    def __init__(self):
        self.clf = None

    def fit(self, X, y):
        self.clf = LogisticRegression().fit(X, y)
        return self

    def transform(self, X):
        if self.clf is None:
            raise NotFittedError()
        return self.clf.predict_proba(X)[:, 1]
