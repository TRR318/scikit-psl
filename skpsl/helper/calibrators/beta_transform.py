import warnings

import numpy as np
from scipy.optimize import minimize_scalar
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.exceptions import NotFittedError
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import column_or_1d, indexable


class BetaTransformer(BaseEstimator, TransformerMixin):
    """
    Adapted from:

    Beta regression model with three parameters introduced in
    Kull, M., Silva Filho, T.M. and Flach, P. Beta calibration: a well-founded
    and easily implemented improvement on logistic calibration for binary
    classifiers. AISTATS 2017.

    Attributes
    ----------
    map_ : array-like, shape (3,)
        Array containing the coefficients of the model (a and b) and the
        midpoint m. Takes the form map_ = [a, b, m]

    lr_ : sklearn.linear_model.LogisticRegression
        Internal logistic regression used to train the model.
    """

    def __init__(self, penalty=None):
        self.penalty = penalty

        self.scaler_ = None
        self.lr_ = None
        self.map_ = None

    def fit(self, X, y, sample_weight=None):
        """Fit the model using X, y as training data.

        Parameters
        ----------
        X : array-like, shape (n_samples,)
            Training data.

        y : array-like, shape (n_samples,)
            Training target.

        sample_weight : array-like, shape = [n_samples] or None
            Sample weights. If None, then samples are equally weighted.

        Returns
        -------
        self : object
            Returns an instance of self.
        """

        self.scaler_ = MinMaxScaler().fit(X.astype(float), y)
        X = self.scaler_.transform(X)
        X = column_or_1d(X)
        y = column_or_1d(y)
        X, y = indexable(X, y)
        warnings.filterwarnings("ignore"),

        df = column_or_1d(X).reshape(-1, 1)
        eps = np.finfo(df.dtype).eps
        df = np.clip(df, eps, 1 - eps)
        y = column_or_1d(y)

        x = np.hstack((df, 1.0 - df))
        x = np.log(x)
        x[:, 1] *= -1

        lr = LogisticRegression(penalty=self.penalty)
        lr.fit(x, y, sample_weight)
        coefs = lr.coef_[0]

        if coefs[0] < 0:
            x = x[:, 1].reshape(-1, 1)
            lr = LogisticRegression(penalty=self.penalty)
            lr.fit(x, y, sample_weight)
            coefs = lr.coef_[0]
            a = None
            b = coefs[0]
        elif coefs[1] < 0:
            x = x[:, 0].reshape(-1, 1)
            lr = LogisticRegression(penalty=self.penalty)
            lr.fit(x, y, sample_weight)
            coefs = lr.coef_[0]
            a = coefs[0]
            b = None
        else:
            a = coefs[0]
            b = coefs[1]
        inter = lr.intercept_[0]

        a_, b_ = a or 0, b or 0
        m = minimize_scalar(
            lambda mh: np.abs(b_ * np.log(1.0 - mh) - a_ * np.log(mh) - inter),
            bounds=[0, 1],
            method="Bounded",
        ).x

        self.map_, self.lr_ = [a, b, m], lr

        return self

    def predict(self, S):
        """Predict new values.

        Parameters
        ----------
        S : array-like, shape (n_samples,)
            Data to predict from.

        Returns
        -------
        S_ : array, shape (n_samples,)
            The predicted values.
        """
        df = column_or_1d(S).reshape(-1, 1)
        eps = np.finfo(df.dtype).eps
        df = np.clip(df, eps, 1 - eps)

        x = np.hstack((df, 1.0 - df))
        x = np.log(x)
        x[:, 1] *= -1
        if self.map_[0] is None:
            x = x[:, 1].reshape(-1, 1)
        elif self.map_[1] is None:
            x = x[:, 0].reshape(-1, 1)

        return self.lr_.predict_proba(x)[:, 1]

    def transform(self, X):
        if self.lr_ is None:
            raise NotFittedError()
        return self.predict(self.scaler_.transform(X.astype(float)))
