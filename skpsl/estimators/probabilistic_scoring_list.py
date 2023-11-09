import logging
from itertools import combinations, product, repeat
from typing import Optional

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.stats import entropy
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.exceptions import NotFittedError
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss

from skpsl.helper import create_optimizer


class _ClassifierAtK(BaseEstimator, ClassifierMixin):
    """
    Internal class for the classifier at stage k of the probabilistic scoring list
    """

    def __init__(
            self,
            features: tuple[int],
            scores: tuple[int],
            initial_thresholds: tuple[Optional[float]],
            threshold_optimizer: callable
    ):
        """

        :param features: tuple of feature indices. used for selecting data from X
        :param scores: tuple of scores, corresponding to the feature indices
        :param initial_thresholds: tuple of thresholds to binarize the feature values
        """
        self.features = features
        self.scores = scores
        self.initial_thresholds = initial_thresholds
        self.threshold_optimizer = threshold_optimizer

        self.thresholds = list(initial_thresholds)
        self.scores_vec = np.array(scores)
        self.logger = logging.getLogger(__name__)
        self.calibrator = None

    def fit(self, X, y) -> "_ClassifierAtK":
        for i, (f, t) in enumerate(zip(self.features, self.initial_thresholds)):
            feature_values = X[:, f]
            is_data_binary = set(np.unique(feature_values).astype(int)) <= {0, 1}
            if t is np.nan and not is_data_binary:
                self.logger.debug(f"feature {f} is non-binary and threshold not set: calculating threshold...")
                # fit optimal threshold
                self.thresholds[i] = self.threshold_optimizer(
                    lambda t_, _: self._expected_entropy(self._compute_total_scores(
                        X=X,
                        features=self.features[: i + 1],
                        scores=self.scores_vec[: i + 1],
                        thresholds=self.thresholds[:i] + [t_]), y
                    ),
                    feature_values,
                )

        total_scores = self._compute_total_scores(X, self.features, self.scores_vec, self.thresholds)

        # compute probabilities
        self.calibrator = IsotonicRegression(
            y_min=0.0, y_max=1.0, increasing=True, out_of_bounds="clip"
        )
        self.calibrator.fit(total_scores, y)

        return self

    def predict(self, X):
        if self.calibrator is None:
            raise NotFittedError()
        return self.predict_proba(X).argmax(axis=1)

    def predict_proba(self, X):
        """
        Predicts the probability for
        """
        if self.calibrator is None:
            raise NotFittedError()
        proba_true = self.calibrator.transform(
            self._compute_total_scores(X, self.features, self.scores_vec, self.thresholds)
        )
        proba = np.vstack([1 - proba_true, proba_true]).T
        return proba

    def score(self, X, y=None, sample_weight=None):
        """
        Calculates the expected entropy of the fitted model
        :param X:
        :param y:
        :param sample_weight:
        :return:
        """
        if self.calibrator is None:
            raise NotFittedError()
        if not self.features:
            p_pos = self.calibrator.transform([[0]])
            return entropy([p_pos, 1 - p_pos], base=2).item()
        total_scores = self._compute_total_scores(X, self.features, self.scores_vec, self.thresholds)
        return self._expected_entropy(total_scores, calibrator=self.calibrator)

    @staticmethod
    def _compute_total_scores(X, features, scores: np.ndarray, thresholds):
        if not features:
            return np.zeros(X.shape[0])
        data = np.array(X)[:, features]
        thresholds = np.array(thresholds)
        thresholds[np.isnan(thresholds)] = 0.5
        return (data > thresholds[None, :]) @ scores

    @staticmethod
    def _expected_entropy(X, y=None, calibrator=None):
        if calibrator is None:
            if y is None:
                raise AttributeError(
                    "_expected_entropy must not be called with both 'calibrator' and 'y' being None"
                )
            calibrator = IsotonicRegression(
                y_min=0.0, y_max=1.0, increasing=True, out_of_bounds="clip"
            )
            calibrator.fit(X, y)
        total_scores, score_freqs = np.unique(X, return_counts=True)
        score_probas = np.array(calibrator.transform(total_scores))
        entropy_values = entropy([score_probas, 1 - score_probas], base=2)
        return np.sum((score_freqs / X.size) * entropy_values)


class ProbabilisticScoringList(BaseEstimator, ClassifierMixin):
    """
    Probabilistic scoring list classifier.
    A probabilistic classifier that greedily creates a PSL selecting one feature at a time
    """

    def __init__(self, score_set: set, entropy_threshold: float = -1, method="bisect"):
        """

        :param score_set: Set score values to be considered. Basically feature weights.
        :param entropy_threshold: Shannon Entropy base 2 threshold after which to stop fitting more stages.
        """
        self.score_set = score_set
        self.entropy_threshold = entropy_threshold
        self.method = method

        self._thresh_optimizer = create_optimizer(method)
        self.logger = logging.getLogger(__name__)
        self.sorted_score_set = np.array(sorted(self.score_set, reverse=True, key=abs))
        self.stage_clfs = []  # type: list[_ClassifierAtK]

    def fit(
            self, X, y, lookahead=1, n_jobs=1,
            predef_features: Optional[np.ndarray] = None,
            predef_scores: Optional[np.ndarray] = None
    ) -> "ProbabilisticScoringList":
        """
        Fits a probabilistic scoring list to the given data

        :param X:
        :param y:
        :param lookahead: steps of look ahead
        :param n_jobs: passed to joblib for parallelization
        :param predef_features:
        :param predef_scores:
        :return: The fitted classifier
        """

        number_features = X.shape[1]
        remaining_features = list(range(number_features))

        if predef_features is not None and len(predef_features) != number_features:
            raise ValueError(
                "Predefined features must be a permutation of all features!"
            )

        stage = 0

        # first
        expected_entropy = self._fit_and_store_clf_at_k(X, y)

        while remaining_features and expected_entropy > self.entropy_threshold:
            stage += 1

            # noinspection PyUnresolvedReferences
            features_to_consider = (
                remaining_features
                if predef_features is None
                else [predef_features[stage - 1]]
            )
            scores_to_consider = (
                self.sorted_score_set
                if predef_scores is None
                else [predef_scores[stage - 1]]
            )

            entropies, f, s, t = zip(
                *Parallel(n_jobs=n_jobs)(
                    delayed(self._optimize)(
                        self.features,
                        f_seq,
                        self.scores,
                        list(s_seq),
                        self.thresholds,
                        self._thresh_optimizer,
                        X,
                        y,
                    )
                    for (f_seq, s_seq) in product(
                        self._gen_lookahead(features_to_consider, lookahead),
                        # cartesian power of scores
                        product(
                            *repeat(
                                scores_to_consider,
                                min(lookahead, len(features_to_consider)),
                            )
                        ),
                    )
                )
            )

            i = np.argmin(entropies)
            remaining_features.remove(f[i])

            expected_entropy = self._fit_and_store_clf_at_k(
                X,
                y,
                self.features + [f[i]],
                self.scores + [s[i]],
                self.thresholds + [t[i]],
            )
        return self

    def predict(self, X, k=-1):
        """
        Predicts a probabilistic scoring list to the given data
        :param X: Dataset to predict the probabilities for
        :param k: Classifier stage to use for prediction
        :return:
        """

        return self.predict_proba(X, k).argmax(axis=1)

    def predict_proba(self, X, k=-1):
        """
        Predicts the probability using the k-th or last classifier
        :param X: Dataset to predict the probabilities for
        :param k: Classifier stage to use for prediction
        :return:
        """
        if not self.stage_clfs:
            raise NotFittedError(
                "Please fit the probabilistic scoring classifier before usage."
            )

        return self.stage_clfs[k].predict_proba(X)

    def score(self, X, y, k=-1, sample_weight=None):
        """
        Calculates the Brier score of the model
        :param X:
        :param y:
        :param k: Classifier stage to use for prediction
        :param sample_weight: ignored
        :return:
        """
        return brier_score_loss(y, self.predict_proba(X, k=k)[:, 1])

    def inspect(self, k=None, feature_names=None) -> pd.DataFrame:
        """
        Returns a dataframe that visualizes the internal model

        :param k: maximum stage to include in the visualization (default: all stages)
        :param feature_names: names of the features.
        :return:
        """
        k = k or len(self.stage_clfs) - 1

        scores = self.stage_clfs[k].scores
        features = self.stage_clfs[k].features
        thresholds = self.stage_clfs[k].thresholds

        all_total_scores = [{0}]
        for i, score in enumerate(scores):
            all_total_scores.append(
                all_total_scores[i]
                | {prev_sum + score for prev_sum in all_total_scores[i]}
            )

        data = []
        for clf, T in zip(self.stage_clfs[: k + 1], all_total_scores):
            a = {t: np.nan for t in all_total_scores[-1]}
            probas = clf.calibrator.transform(np.array(list(T)))
            a.update(dict(zip(T, probas)))
            data.append(a)

        df = pd.DataFrame(data)
        df = df[sorted(df.columns)]
        df.columns = [f"T = {t_}" for t_ in df.columns]
        df.insert(0, "Score", np.array([np.nan] + list(scores)))
        if feature_names is not None:
            if len(feature_names) != len(features):
                raise ValueError(
                    f"Passed feature names are of incorrect length! Passed {len(feature_names)}, expected {len(features)}."
                )
            feature_names = [feature_names[i] for i in features]
            df.insert(0, "Feature", [np.nan] + feature_names[:k])
        else:
            df.insert(
                0,
                "Feature Index",
                [np.nan] + features[:k],
            )
        if not all(np.isnan(t) for t in thresholds):
            df.insert(
                0,
                "Threshold",
                [np.nan]
                + [(np.nan if np.isnan(t) else f">{t:.4f}") for t in thresholds],
            )
        return df.reset_index(names=["Stage"])

    @property
    def features(self):
        return self.stage_clfs[-1].features if self.stage_clfs else []

    @property
    def scores(self):
        return self.stage_clfs[-1].scores if self.stage_clfs else []

    @property
    def thresholds(self):
        return self.stage_clfs[-1].thresholds if self.stage_clfs else []

    def _fit_and_store_clf_at_k(self, X, y, f=None, s=None, t=None):
        f, s, t = f or [], s or [], t or []
        k_clf = _ClassifierAtK(features=f, scores=s, initial_thresholds=t,
                               threshold_optimizer=self._thresh_optimizer).fit(X, y)
        self.stage_clfs.append(k_clf)
        return k_clf.score(X)

    @staticmethod
    def _optimize(
            features, feature_extension, scores, score_extension, thresholds, optimizer, X, y
    ):
        clf = _ClassifierAtK(
            features=features + feature_extension,
            scores=scores + score_extension,
            initial_thresholds=thresholds + [np.nan] * len(feature_extension),
            threshold_optimizer=optimizer
        ).fit(X, y)
        return (
            clf.score(X),
            feature_extension[0],
            score_extension[0],
            clf.thresholds[len(features)],
        )

    @staticmethod
    def _gen_lookahead(list_, lookahead):
        return combinations(list_, min(lookahead, len(list_)))


if __name__ == "__main__":
    from sklearn.datasets import make_classification
    from sklearn.model_selection import cross_val_score

    # Generating synthetic data with continuous features and a binary target variable
    X_, y_ = make_classification(random_state=42)

    clf_ = ProbabilisticScoringList({-1, 1, 2})
    print("Brier score:", cross_val_score(clf_, X_, y_, cv=5).mean())
