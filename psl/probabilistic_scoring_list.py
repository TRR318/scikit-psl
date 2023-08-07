import logging

import numpy as np
from scipy.stats import entropy as stats_entropy
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.exceptions import NotFittedError
from sklearn.isotonic import IsotonicRegression


class _ClassifierAtK(BaseEstimator, ClassifierMixin):
    """Internal class for the classifier at stage k of the probabilistic scoring list
    This allows for easy calibration using the SKLearn interface
    """

    def __init__(self, scores, features) -> None:
        self.logger = logging.getLogger(__name__)
        self.scores = np.array(scores)
        self.features = features
        self.score_sums = {}
        self.probabilities = {}
        self.entropy = None
        self.calibrator = None

    def fit(self, X, y):
        n = X.shape[0]
        relevant_scores = self._relevant_scores(X)

        # compute all possible total scores using subset-summation
        self.score_sums = {0}
        for score in self.scores:
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
        return np.sum(X[:, self.features] * self.scores, axis=1)


class ProbabilisticScoringList(BaseEstimator, ClassifierMixin):
    def __init__(self, score_set, entropy_threshold=-1):
        """ IMPORTANT: Shannon entropy is calculated with respect to base 2
        """
        self.logger = logging.getLogger(__name__)
        self.score_set = score_set
        self.sorted_score_set = sorted(self.score_set, reverse=True)
        self.entropy_threshold = entropy_threshold
        self.X = None
        self.y = None
        self.scores = []
        self.features = []
        self.total_scores_at_k = []
        self.probabilities_at_k = []
        self.classifier_at_k = []
        self.entropy_at_k = []
        self._ClassifierAtK = _ClassifierAtK

    def fit(self, X, y, predefined_features=None, predefined_scores=None):
        """Fits a probabilistic scoring list to the given data
        """

        number_features = X.shape[1]
        remaining_feature_indices = list(range(number_features))

        if predefined_features is not None and len(predefined_features) != number_features:
            raise ValueError("Predefined features must be a permutation of all features!")

        self.X = X
        self.y = y

        stage = 0

        self.features = []
        self.scores = []

        # first 
        temp_classifier_at_k = self._ClassifierAtK(features=[], scores=[])
        temp_classifier_at_k.fit(X, y)
        temp_expected_entropy = temp_classifier_at_k.entropy
        self.classifier_at_k.append(temp_classifier_at_k)
        expected_entropy = temp_expected_entropy

        while remaining_feature_indices and expected_entropy > self.entropy_threshold:
            stage += 1
            classifier_at_k = None

            # try all features and possible scores
            fk, sk = remaining_feature_indices[0], self.sorted_score_set[-1]
            current_expected_entropy = np.inf

            if predefined_features is None:
                features_to_consider = remaining_feature_indices
            else:
                features_to_consider = [predefined_features[stage - 1]]

            scores_to_consider = self.sorted_score_set if predefined_scores is None else [predefined_scores[stage - 1]]

            for f in features_to_consider:
                cand_features = self.features + [f]
                for s in scores_to_consider:
                    temp_scores = np.array(self.scores + [s])
                    temp_classifier_at_k = self._ClassifierAtK(features=cand_features, scores=temp_scores)
                    temp_classifier_at_k.fit(X, y)
                    temp_expected_entropy = temp_classifier_at_k.entropy
                    # self.logger.info(f"feature {f} scores {s} entropy {temp_expected_entropy}")
                    if temp_expected_entropy < current_expected_entropy:
                        current_expected_entropy = temp_expected_entropy
                        fk, sk = f, s
                        classifier_at_k = temp_classifier_at_k

            expected_entropy = current_expected_entropy
            self.classifier_at_k.append(classifier_at_k)

            remaining_feature_indices.remove(fk)

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
        if not self.classifier_at_k:
            raise NotFittedError("Please fit the probabilistic scoring classifier before usage.")

        return self.classifier_at_k[k].predict_proba(X)


if __name__ == '__main__':
    from sklearn.datasets import make_classification
    from sklearn.model_selection import cross_val_score

    # Generating synthetic data with continuous features and a binary target variable
    X, y = make_classification(n_samples=100, n_features=10, n_informative=10, n_redundant=0, random_state=42)
    X = (X > .5).astype(int)

    clf = ProbabilisticScoringList([-1, 1, 2])
    print(cross_val_score(clf, X, y, cv=5))

