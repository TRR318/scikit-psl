import logging
from functools import partial
from itertools import combinations, product, repeat
from typing import List, Union

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.stats import entropy as stats_entropy
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.exceptions import NotFittedError
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss

from skpsl.data.util import logarithmic_optimizer


class _ClassifierAtK(BaseEstimator, ClassifierMixin):
    """
    Internal class for the classifier at stage k of the probabilistic scoring list
    """

    def __init__(self, features: tuple[int], scores: tuple[int], initial_thresholds: tuple[Union[float, None]]):
        """

        :param features: tuple of feature indices. used for selecting data from X
        :param scores: tuple of scores, corresponding to the feature indices
        :param initial_thresholds: tuple of thresholds to binarize the feature values
        """
        self.features = features
        self.scores = scores
        self.initial_thresholds = initial_thresholds

        self.thresholds = list(initial_thresholds)
        self.scores_vec = np.array(scores)
        self.logger = logging.getLogger(__name__)
        self.calibrator = None

    def fit(self, X, y) -> "_ClassifierAtK":
        for i, (f, s, t) in enumerate(zip(self.features, self.scores, self.initial_thresholds)):
            feature_values = X[:, f]
            is_data_binary = set(np.unique(feature_values).astype(int)) == {0, 1}
            self.logger.debug(f"feature {f} is non-binary, calculating threshold")
            if t is None:
                if not is_data_binary:
                    # fit optimal threshold
                    self.thresholds[i] = logarithmic_optimizer(
                        partial(self._expected_entropy, thresholds_prefix=self.thresholds[:i],
                                X=X,
                                features=self.features[:i + 1], scores=self.scores_vec[:i + 1], y=y),
                        feature_values)
                else:
                    # TODO this can be more efficient. There needs to be no actuall thresholding for truely binary data.
                    self.thresholds[i] = .5

        scores = self._scores_per_binarized_record(X, self.features, self.scores_vec, self.thresholds)

        # compute probabilities
        self.calibrator = IsotonicRegression(y_min=0.0, y_max=1.0, increasing=True, out_of_bounds="clip")
        self.calibrator.fit(scores, y)

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
            self._scores_per_binarized_record(X, self.features, self.scores_vec, self.thresholds))
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
            return stats_entropy(self.calibrator.transform([[0]]))
        return self._expected_entropy(self.thresholds[-1], None, self.thresholds[:-1], X,
                                      self.features, self.scores_vec, self.calibrator)

    @staticmethod
    def _expected_entropy(threshold, data_slice,
                          thresholds_prefix, X,
                          features, scores: np.ndarray,
                          calibrator=None, y=None):
        thresholds = thresholds_prefix + [threshold]
        scores = _ClassifierAtK._scores_per_binarized_record(X, features, scores, thresholds)
        if calibrator is None:
            calibrator = IsotonicRegression(y_min=0.0, y_max=1.0, increasing=True, out_of_bounds="clip")
            calibrator.fit(scores, y)
        total_scores, score_freqs = np.unique(scores, return_counts=True)
        score_probas = np.array(calibrator.transform(total_scores))
        entropy_values = stats_entropy([score_probas, 1 - score_probas], base=2)
        return np.sum((score_freqs / scores.size) * entropy_values)

    @staticmethod
    def _scores_per_binarized_record(X, features, scores: np.ndarray, thresholds):
        data = np.array(X)[:, features]
        thresholds = np.array(thresholds)[None, :]
        return (data > thresholds) @ scores


class ProbabilisticScoringList(BaseEstimator, ClassifierMixin):
    """
    Probabilistic scoring list classifier.
    A probabilistic classifier that greedily creates a PSL selecting one feature at a time
    """

    def __init__(self, score_set: set, entropy_threshold: float = -1):
        """

        :param score_set: Set score values to be considered. Basically feature weights.
        :param entropy_threshold: Shannon Entropy base 2 threshold after which to stop fitting more stages.
        """
        self.score_set = score_set
        self.entropy_threshold = entropy_threshold

        self.logger = logging.getLogger(__name__)
        self.sorted_score_set = sorted(self.score_set, reverse=True)
        self._stage_clf = _ClassifierAtK
        self.stage_clfs = []  # type: List[_ClassifierAtK]

    def fit(self, X, y, lookahead=1, n_jobs=1, predef_features=None, predef_scores=None) -> "ProbabilisticScoringList":
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
            raise ValueError("Predefined features must be a permutation of all features!")

        stage = 0

        # first 
        expected_entropy = self._fit_and_store_clf_at_k(X, y)

        while remaining_features and expected_entropy > self.entropy_threshold:
            stage += 1

            # noinspection PyUnresolvedReferences
            features_to_consider = remaining_features if predef_features is None else [predef_features[stage - 1]]
            scores_to_consider = self.sorted_score_set if predef_scores is None else [predef_scores[stage - 1]]

            entropies, f, s, t = zip(*Parallel(n_jobs=n_jobs)(
                delayed(self._optimize)(self.features, f_seq, self.scores, list(s_seq), self.thresholds,
                                        self._stage_clf, X, y)
                for (f_seq, s_seq) in product(
                    self._gen_lookahead(features_to_consider, lookahead),
                    # cartesian power of scores
                    product(*repeat(scores_to_consider, min(lookahead, len(features_to_consider))))
                )
            ))

            i = np.argmin(entropies)
            remaining_features.remove(f[i])

            expected_entropy = self._fit_and_store_clf_at_k(X, y,
                                                            self.features + [f[i]],
                                                            self.scores + [s[i]],
                                                            self.thresholds + [t[i]])
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
            raise NotFittedError("Please fit the probabilistic scoring classifier before usage.")

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

        pdfs = [clf.calibrator for clf in self.stage_clfs[:k + 1]]
        scores = np.array(self.stage_clfs[k].scores)

        all_total_scores = [[0]]
        total_scores = {0}
        for score in scores:
            total_scores |= {prev_sum + score for prev_sum in total_scores}
            all_total_scores.append(np.array(sorted(total_scores)))

        data = []
        for pdf, T in zip(pdfs, all_total_scores):
            a = {t: np.nan for t in all_total_scores[-1]}
            a.update({t: v for t, v in zip(T, pdf.transform(T))})
            data.append(a)

        df = pd.DataFrame(data)
        df.columns = [f"T = {t_}" for t_ in df.columns]
        df.insert(0, "Score", np.array([np.nan] + list(self.stage_clfs[k].scores)))
        if feature_names is not None:
            df.insert(0, "Feature", [np.nan] + feature_names[:k] + [np.nan] * (k - len(feature_names)))
        if all(t is not None for t in self.stage_clfs[k].thresholds):
            df.insert(0, "Threshold", [np.nan] + [f">{t:.4f}" for t in self.stage_clfs[k].thresholds])
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
        k_clf = self._stage_clf(features=f, scores=s, initial_thresholds=t).fit(X, y)
        self.stage_clfs.append(k_clf)
        return k_clf.score(X)

    @staticmethod
    def _optimize(features, feature_extension, scores, score_extension, thresholds, clfcls, X, y):
        clf = clfcls(features=features + feature_extension, scores=scores + score_extension,
                     initial_thresholds=thresholds + [None] * len(feature_extension)).fit(
            X, y)
        return clf.score(X), feature_extension[0], score_extension[0], clf.thresholds[len(features)]

    @staticmethod
    def _gen_lookahead(list_, lookahead):
        # generate sequences of shortening lookaheads (because combinations returns empty list if len(list) < l)
        combination_seqs = ([list(tup) for tup in combinations(list_, _l)] for _l in range(lookahead, 0, -1))
        # get first non-empty sequence
        seqs = next((seq for seq in combination_seqs if seq))
        return seqs


if __name__ == '__main__':
    from sklearn.datasets import make_classification
    from sklearn.model_selection import cross_val_score

    # Generating synthetic data with continuous features and a binary target variable
    X_, y_ = make_classification(random_state=42)

    clf_ = ProbabilisticScoringList({-1, 1, 2})
    print("Brier score:", cross_val_score(clf_, X_, y_, cv=5).mean())
