import logging

import numpy as np
from scipy.stats import entropy as stats_entropy
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.exceptions import NotFittedError
from sklearn.isotonic import IsotonicRegression


class _ClassifierAtK(BaseEstimator, ClassifierMixin):
    """
    Internal class for the classifier at stage k of the probabilistic scoring list
    """

    def __init__(self, scores, features):
        self.scores = scores
        self.features = features

        self.logger = logging.getLogger(__name__)
        self.scores_vec = np.array(scores)
        self.score_sums = {}
        self.probabilities = {}
        self.entropy = None
        self.calibrator = None

    def fit(self, X, y) -> "_ClassifierAtK":
        n = X.shape[0]
        relevant_scores = self._relevant_scores(X)

        # compute all possible total scores using subset-summation
        self.score_sums = {0}
        for score in self.scores_vec:
            self.score_sums |= {prev_sum + score for prev_sum in self.score_sums}

        # calibrate probabilities
        self.calibrator = IsotonicRegression(y_min=0.0, y_max=1.0, increasing=True, out_of_bounds="clip")
        self.calibrator.fit(relevant_scores, y)

        # compute calibrated probabilities
        sigmaK = np.array(sorted(self.score_sums))
        cal_ps = self.calibrator.predict(sigmaK)

        # set calibrated probabilities
        self.probabilities = {T: p for T, p in zip(sigmaK, cal_ps)}
        self.entropy = 0
        for ti, pi in self.probabilities.items():
            Ni = np.count_nonzero(relevant_scores == ti)
            Hi = stats_entropy([pi, 1 - pi], base=2)
            self.entropy += (Ni / n) * Hi

        return self

    def predict(self, X):
        return self.predict_proba(X).argmax(axis=1)

    def predict_proba(self, X):
        """Predicts the probability for
        """
        proba_true = np.vectorize(self.probabilities.get)(self._relevant_scores(X))
        proba = np.vstack([1 - proba_true, proba_true]).T
        return proba

    # Helper functions
    def _relevant_scores(self, X):
        return np.sum(X[:, self.features] * self.scores_vec, axis=1)


class ProbabilisticScoringList(BaseEstimator, ClassifierMixin):
    """
    Probabilistic scoring list classifier. A probabilistic classifier that greedily creates a PSL selecting one feature at a time
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

    def fit(self, X, y, predef_features=None, predef_scores=None) -> "ProbabilisticScoringList":
        """Fits a probabilistic scoring list to the given data
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

            # try all features and possible scores
            curr_stage_clf = None
            fk, sk = remaining_features[0], self.sorted_score_set[-1]
            current_expected_entropy = np.inf

            features_to_consider = remaining_features if predef_features is None else [predef_features[stage - 1]]
            scores_to_consider = self.sorted_score_set if predef_scores is None else [predef_scores[stage - 1]]

            for f in features_to_consider:
                cand_features = self.features + [f]
                for s in scores_to_consider:
                    tmp_stage_clf = self._stage_clf(features=cand_features, scores=np.array(self.scores + [s]))
                    tmp_stage_clf.fit(X, y)
                    temp_expected_entropy = tmp_stage_clf.entropy

                    self.logger.info(f"feature {f} scores {s} entropy {temp_expected_entropy}")
                    if temp_expected_entropy < current_expected_entropy:
                        current_expected_entropy = temp_expected_entropy
                        fk, sk = f, s
                        curr_stage_clf = tmp_stage_clf

            expected_entropy = current_expected_entropy
            self.stage_clfs.append(curr_stage_clf)

            remaining_features.remove(fk)

            self.features.append(fk)
            self.scores.append(sk)
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


if __name__ == '__main__':
    from sklearn.datasets import make_classification
    from sklearn.model_selection import cross_val_score

    # Generating synthetic data with continuous features and a binary target variable
    X, y = make_classification(n_samples=100, n_features=10, n_informative=10, n_redundant=0, random_state=42)
    X = (X > .5).astype(int)

    clf = ProbabilisticScoringList([-1, 1, 2])
    print(cross_val_score(clf, X, y, cv=5))
