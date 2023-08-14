import logging
from functools import partial
from typing import Tuple

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import entropy
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.exceptions import NotFittedError
from sklearn.preprocessing import MinMaxScaler

logger = logging.getLogger()


class MinEntropyBinarizer(BaseEstimator, TransformerMixin, auto_wrap_output_keys=None):
    def __init__(self):
        self.threshs = None

    def fit(self, X, y=None):
        self.threshs = np.apply_along_axis(lambda x: MinEntropyBinarizer._binarize(x, y=y)[0], 0, X)
        return self

    def transform(self, X):
        if self.threshs is None:
            raise NotFittedError()
        return np.apply_along_axis(lambda x: MinEntropyBinarizer._binarize(x[1:], thresh=x[0])[1], 0,
                                   np.vstack([self.threshs.reshape(1, -1), X]))

    def inspect(self, feature_names=None) -> pd.DataFrame:
        """
        Returns a dataframe that visualizes the internal model
        """
        k = len(self.threshs) 
        df = pd.DataFrame(columns=["Threshold"], data=self.threshs)
        if feature_names is not None:
            df.insert(0, "Feature", feature_names[:k] + [np.nan] * (k - len(feature_names)))
        return df


    @staticmethod
    def _cut_entropy(x: np.ndarray, y: np.ndarray, thresh: float) -> float:
        """
        https://www.ijcai.org/Proceedings/93-2/Papers/022.pdf
        @param x: one-dimensional float array of the feature variable
        @param y: one-dimensional float array of the target variable
        @param thresh: scalar or scalar array
        @return: combined entropy as float
        """
        thresh = np.array(thresh).squeeze()
        s1 = y[x < thresh]
        s2 = y[x >= thresh]
        s1_freqs = np.unique(s1, return_counts=True)[1] / s1.size
        s2_freqs = np.unique(s2, return_counts=True)[1] / s2.size

        return (s1.size / x.size * entropy(s1_freqs, base=2) +
                s2.size / x.size * entropy(s2_freqs, base=2))

    @staticmethod
    def _binarize(x: np.ndarray, y: np.ndarray = None, thresh=None) -> Tuple[float, np.ndarray]:
        """
        @param x: one-dimensional float array of the feature variable
        @param y: one-dimensional float array of the target variable
        @return: threshold, one-dimensional integer array of binarized feature
        """
        uniq = np.unique(x)
        if uniq.size < 3:
            thresh = np.min(uniq) + (np.max(uniq) - np.min(uniq)) / 2
        if thresh is None:
            scaler = MinMaxScaler()
            x_ = scaler.fit_transform(x.reshape(-1, 1)).squeeze()  # type:np.ndarray
            thresh = minimize(partial(MinEntropyBinarizer._cut_entropy, x_, y), x.mean(), method="Nelder-Mead",
                              options=dict(initial_simplex=[[0], [1]])).x.squeeze().item()
            thresh = scaler.inverse_transform([[thresh]]).squeeze()
            logger.debug(thresh)
        return thresh, (x >= thresh).astype(int)


if __name__ == '__main__':
    from sklearn.datasets import make_classification

    logging.basicConfig()
    logger.setLevel(logging.DEBUG)
    # Generating synthetic data with continuous features and a binary target variable
    X, y = make_classification(n_samples=5, n_features=2, n_informative=2, n_redundant=0, random_state=42)

    print(np.hstack([X, y.reshape(-1, 1)]))
    print(MinEntropyBinarizer().fit_transform(X, y))

    x1 = np.linspace(0, 100, 10)
    y = (x1 > 30).astype(int)
    X = x1.reshape(-1, 1)
    print(np.hstack([X, y.reshape(-1, 1)]))
    print(MinEntropyBinarizer().fit_transform(X, y))

    X = np.random.randint(0, 2, size=(10, 3))
    y = np.random.randint(0, 2, size=(10,))
    print(np.hstack([X, y.reshape(-1, 1)]))
    print(X == MinEntropyBinarizer().fit_transform(X, y))

    x1 = np.linspace(0, 100, 10)
    y = (x1 > 30).astype(int)
    X = x1.reshape(-1, 1)
    print(np.hstack([X, y.reshape(-1, 1)]))
    print(MinEntropyBinarizer().fit(X, y).transform(X))

    x1 = np.linspace(0, 100, 10)
    y = (x1 > 30).astype(int)
    X = x1.reshape(-1, 1)
    print(np.hstack([X, y.reshape(-1, 1)]))
    print(MinEntropyBinarizer().fit(X, y).inspect())

    X, y = make_classification(n_samples=50, n_features=3, n_informative=2, n_redundant=0, random_state=42)
    print(np.hstack([X, y.reshape(-1, 1)]))
    print(MinEntropyBinarizer().fit(X, y).inspect(feature_names=["width", "height"]))
