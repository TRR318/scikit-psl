import logging
from functools import partial

import numpy as np
import pandas as pd
from scipy.stats import entropy
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.exceptions import NotFittedError

from skpsl.data.util import logarithmic_minimizer

logger = logging.getLogger()


class MinEntropyBinarizer(BaseEstimator, TransformerMixin, auto_wrap_output_keys=None):
    def __init__(self):
        self.threshs = None

    def fit(self, X, y=None):
        def binarize(x):
            uniq = np.unique(x)
            if uniq.size < 3:
                return np.min(uniq) + (np.max(uniq) - np.min(uniq)) / 2
            return logarithmic_minimizer(partial(MinEntropyBinarizer._cut_entropy, y=y), x)

        self.threshs = np.apply_along_axis(binarize, 0, X)
        return self

    def transform(self, X):
        if self.threshs is None:
            raise NotFittedError()
        return (np.array(X) >= self.threshs).astype(int)

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
    def _cut_entropy(thresh: float, x: np.array, y: np.array) -> float:
        """
        https://www.ijcai.org/Proceedings/93-2/Papers/022.pdf

        :param x: one-dimensional float array of the feature variable
        :param y: one-dimensional float array of the target variable
        :param thresh: scalar
        :return: combined entropy as float
        """
        mask = x < np.array(thresh).squeeze()
        s1, s2 = y[mask], y[~mask]
        _, s1_freqs = np.unique(s1, return_counts=True)
        _, s2_freqs = np.unique(s2, return_counts=True)

        return (s1.size * entropy(s1_freqs, base=2) +
                s2.size * entropy(s2_freqs, base=2)) / x.size


if __name__ == '__main__':
    from sklearn.datasets import make_classification

    logging.basicConfig()
    logger.setLevel(logging.DEBUG)

    x1 = np.linspace(0, 100, 10)
    y_ = (x1 > 30).astype(int)
    X_ = x1.reshape(-1, 1)
    print(np.hstack([X_, y_.reshape(-1, 1)]))
    print(MinEntropyBinarizer().fit(X_, y_).inspect())

    X_, y_ = make_classification(n_samples=50, n_features=3, n_informative=2, n_redundant=0, random_state=42)
    print(np.hstack([X_, y_.reshape(-1, 1)]))
    print(MinEntropyBinarizer().fit(X_, y_).inspect(feature_names=["width", "height"]))
