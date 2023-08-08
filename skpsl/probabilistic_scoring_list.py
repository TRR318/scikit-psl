import logging
from itertools import combinations, product, repeat

import numpy as np
from joblib import Parallel, delayed
from scipy.stats import entropy as stats_entropy
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.exceptions import NotFittedError
from sklearn.isotonic import IsotonicRegression


class _ClassifierAtK(BaseEstimator, ClassifierMixin):
    """
    Internal class for the classifier at stage k of the probabilistic scoring list
    """

    def __init__(self, features, scores):
        self.features = features
        self.scores = scores

        self.logger = logging.getLogger(__name__)
        self.scores_vec = np.array(scores)
        self.probabilities = {}
        self.entropy = None

    def fit(self, X, y) -> "_ClassifierAtK":
        scores = self._scores_per_record(X)
        n = scores.size

        # compute all possible total scores using subset-summation
        total_scores = {0}
        for score in self.scores_vec:
            total_scores |= {prev_sum + score for prev_sum in total_scores}
        total_scores = np.array(sorted(total_scores))

        # compute probabilities
        calibrator = IsotonicRegression(y_min=0.0, y_max=1.0, increasing=True, out_of_bounds="clip")
        calibrator.fit(scores, y)
        self.probabilities = {T: p for T, p in zip(total_scores, calibrator.transform(total_scores))}

        # TODO. this should actually be inside of a score function. the actual fitting is finished at this point
        total_scores, score_freqs = np.unique(scores, return_counts=True)
        score_probas = np.array([self.probabilities[ti] for ti in total_scores])
        entropy_values = stats_entropy([score_probas, 1 - score_probas], base=2)
        self.entropy = np.sum((score_freqs / n) * entropy_values)

        return self

    def predict(self, X):
        return self.predict_proba(X).argmax(axis=1)

    def predict_proba(self, X):
        """Predicts the probability for
        """
        scores = self._scores_per_record(X)
        proba_true = np.empty_like(scores, dtype=float)
        for total_score in np.unique(scores):
            proba_true[scores == total_score] = self.probabilities[total_score]
        proba = np.vstack([1 - proba_true, proba_true]).T
        return proba

    # Helper functions
    def _scores_per_record(self, X):
        return X[:, self.features] @ self.scores_vec


class ProbabilisticScoringList(BaseEstimator, ClassifierMixin):
    """
    Probabilistic scoring list classifier.
    A probabilistic classifier that greedily creates a PSL selecting one feature at a time
    """

    def __init__(self, score_set, entropy_threshold=-1):
        """ IMPORTANT: Shannon entropy is calculated with respect to base 2
        """
        self.score_set = score_set
        self.entropy_threshold = entropy_threshold

        self.logger = logging.getLogger(__name__)
        self.sorted_score_set = sorted(self.score_set, reverse=True)
        self.X = None
        self.y = None
        self.scores = []
        self.features = []
        self.total_scores_at_k = []
        self.probabilities_at_k = []
        self.stage_clfs = []
        self.entropy_at_k = []
        self._stage_clf = _ClassifierAtK

    def fit(self, X, y, l=1, n_jobs=1, predef_features=None, predef_scores=None) -> "ProbabilisticScoringList":
        """
        Fits a probabilistic scoring list to the given data

        :param X:
        :param y:
        :param l: steps of look ahead
        :param n_jobs: passed to joblib for parallelization
        :param predef_features:
        :param predef_scores:
        :return:
        """

        number_features = X.shape[1]
        remaining_features = list(range(number_features))

        if predef_features is not None and len(predef_features) != number_features:
            raise ValueError("Predefined features must be a permutation of all features!")

        self.X = X
        self.y = y

        self.features = []
        self.scores = []

        stage = 0

        # first 
        curr_stage_clf = self._stage_clf(features=[], scores=[])
        curr_stage_clf.fit(X, y)
        self.stage_clfs.append(curr_stage_clf)
        expected_entropy = curr_stage_clf.entropy

        while remaining_features and expected_entropy > self.entropy_threshold:
            stage += 1

            features_to_consider = remaining_features if predef_features is None else [predef_features[stage - 1]]
            scores_to_consider = self.sorted_score_set if predef_scores is None else [predef_scores[stage - 1]]

            clfs, entropies, f, s = zip(*Parallel(n_jobs=n_jobs)(
                delayed(self._optimize)(self.features, f_seq, self.scores, list(s_seq), self._stage_clf, X, y)
                for (f_seq, s_seq) in product(
                    self._gen_lookahead(features_to_consider, l),
                    # cartesian power of scores
                    product(*repeat(scores_to_consider, min(l, len(features_to_consider))))
                )
            ))

            i = np.argmin(entropies)

            expected_entropy = entropies[i]
            self.stage_clfs.append(clfs[i])
            remaining_features.remove(f[i])
            self.features.append(f[i])
            self.scores.append(s[i])
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

    @staticmethod
    def _optimize(features, feature_extension, scores, score_extension, clfcls, X, y):
        clf = clfcls(features=features + feature_extension, scores=scores + score_extension).fit(X, y)
        return clf, clf.entropy, feature_extension[0], score_extension[0]

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
    X, y = make_classification(random_state=42)
    X = (X > .5).astype(int)

    clf = ProbabilisticScoringList([-1, 1, 2])
    print(cross_val_score(clf, X, y, cv=5))
