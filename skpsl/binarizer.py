import logging
from typing import Tuple

import numpy as np
import pandas as pd
from scipy.stats import entropy
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.exceptions import NotFittedError
from sklearn.preprocessing import MinMaxScaler
from sortedcontainers import SortedSet

logger = logging.getLogger()


class MinEntropyBinarizer(BaseEstimator, TransformerMixin, auto_wrap_output_keys=None):
    def __init__(self):
        self.threshs = None

    def fit(self, X, y=None):
        self.threshs = np.apply_along_axis(lambda x: MinEntropyBinarizer._binarize(x, y=y), 0, X)
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
    def _cut_entropy(x: np.ndarray, y: np.ndarray, thresh: float) -> float:
        """
        https://www.ijcai.org/Proceedings/93-2/Papers/022.pdf
        @param x: one-dimensional float array of the feature variable
        @param y: one-dimensional float array of the target variable
        @param thresh: scalar or scalar array
        @return: combined entropy as float
        """
        mask = x < np.array(thresh).squeeze()
        s1, s2 = y[mask], y[~mask]
        _, s1_freqs = np.unique(s1, return_counts=True)
        _, s2_freqs = np.unique(s2, return_counts=True)

        return (s1.size * entropy(s1_freqs, base=2) +
                s2.size * entropy(s2_freqs, base=2)) / x.size

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
            x_scaled = scaler.fit_transform(x.reshape(-1, 1)).squeeze()
            values = np.sort(np.unique(x_scaled))
            # maybe -inf and +inf are not neccesary, but better safe (authors where not able to proof optimality otherwise)
            # if you have a proof, than please send a PR with the proof
            # as the data is scaled to [0,1] -1 and 2 are effectively -inf and + inf. otherwise inverse transform does not work
            cuts = np.concatenate([[-1], (values[:-1] + values[1:]) / 2, [2]])
            minimal_points = set()
            min_ = np.inf
            thresh = None
            evaluated = SortedSet()
            to_evaluate = {0, cuts.size - 1}

            while to_evaluate:
                # evaluate points
                while to_evaluate:
                    k = to_evaluate.pop()
                    evaluated.add(k)
                    entropy = MinEntropyBinarizer._cut_entropy(x_scaled, y, cuts[k])
                    if entropy < min_:
                        min_ = entropy
                        thresh = cuts[k]
                        minimal_points = {k}
                    elif entropy == min_:
                        minimal_points.add(k)

                # calculate new points to evaluate
                for k in minimal_points:
                    k_pos = evaluated.index(k)
                    candidates = set()
                    for offset in [-1, 1]:
                        try:
                            candidates.add(k + (evaluated[k_pos + offset] - k) // 2)
                        except IndexError:
                            pass
                    to_evaluate.update(candidates - evaluated)
            thresh = scaler.inverse_transform([[thresh]]).item()

        return thresh


if __name__ == '__main__':
    from sklearn.datasets import make_classification

    logging.basicConfig()
    logger.setLevel(logging.DEBUG)

    x1 = np.linspace(0, 100, 10)
    y = (x1 > 30).astype(int)
    X = x1.reshape(-1, 1)
    print(np.hstack([X, y.reshape(-1, 1)]))
    print(MinEntropyBinarizer().fit(X, y).inspect())

    X, y = make_classification(n_samples=50, n_features=3, n_informative=2, n_redundant=0, random_state=42)
    print(np.hstack([X, y.reshape(-1, 1)]))
    print(MinEntropyBinarizer().fit(X, y).inspect(feature_names=["width", "height"]))
