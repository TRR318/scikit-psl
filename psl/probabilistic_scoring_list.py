from itertools import chain, combinations

import numpy as np
from scipy.stats import entropy
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.isotonic import IsotonicRegression


class _ClassifierAtK:
    """Internal class for the classifier at stage k of the probabilistic scoring list
    This allows for easy calibration using the SKLearn interface 
    """

    def __init__(self, scores, features) -> None:
        self._estimator_type = "classifier"
        # self.logger = logging.getLogger(__name__)
        self.features = features
        self.scores = scores
        self.total_scores = []
        self.entropy = None
        self.probabilities = {}
        self.calibrator = None

    def fit(self, X, y):
        # train classifier
        self.total_scores = self._compute_possible_total_scores()
        temp_X = X[:, self.features]

        self.score_vector = np.array(self.scores)
        base_probabilities = self._compute_probability_distribution(temp_X, y, self.score_vector, self.total_scores)

        T_xi = np.sum((temp_X * self.score_vector), axis=1)
        Ts = np.array(list(base_probabilities.keys()))

        # calibrate probabilities TODO check this carefully
        calibrator = IsotonicRegression(y_min=0.0, y_max=1.0, increasing=True, out_of_bounds="clip")
        calibrator.fit(T_xi, y)
        self.calibrator = calibrator

        # compute calibrated probabilities
        sigmaK = np.array(list(self.total_scores))
        cal_ps = self.calibrator.predict(sigmaK)

        # set calibrated probabilities
        self.probabilities = {T: p for T, p in zip(sigmaK, cal_ps)}
        self.entropy = self._compute_expected_entropy(X)
        return self

    def predict(self, X):
        probas = self.predict_proba(X)
        return probas.argmax(axis=1)

    def predict_proba(self, X):
        """Predicts the probability for 
        """
        S = np.array(self.scores)
        temp_X = X[:, self.features]
        mat_Ni = temp_X * S
        current_scores = np.sum(mat_Ni, axis=1)
        proba_true = np.vectorize(self.probabilities.get)(current_scores)
        proba = np.vstack([1 - proba_true.T, proba_true.T])
        return proba.T

    # Helper functions
    def _estimate_probability(self, X, y, S, T):
        """Private helper to estimate probability (\tilde{p} in the paper)
        """
        return self._get_Pi(X, y, S, T) / self._get_Ni(X, S, T)

    def _compute_possible_total_scores(self):
        """Private helper to compute the set of all possible total scores
        """
        possible_values = set()
        for subset in powerset(self.scores):
            possible_values.add(sum(subset))
        return possible_values

    def _get_Ni(self, X, S, T):
        mat_Ni = X * S
        Ts_Ni = np.sum(mat_Ni, axis=1)
        Ni = np.count_nonzero(Ts_Ni == T)
        return Ni

    def _get_Pi(self, X, y, S, T):
        X_true = X[y > 0]
        mat_Pi = X_true * S
        Ts_Pi = np.sum(mat_Pi, axis=1)
        Pi = np.count_nonzero(Ts_Pi == T)
        return Pi

    def _compute_probability_distribution(self, X, y, score_vector, total_scores):
        probability_distribution = {}
        for ti in total_scores:
            Ni = self._get_Ni(X, score_vector, ti)
            if Ni == 0:
                continue
            N = X.shape[0]
            pi = self._estimate_probability(X, y, score_vector, ti)
            probability_distribution[ti] = pi
        return probability_distribution

    def _compute_expected_entropy(self, X):
        temp_X = X[:, self.features]
        sum = 0
        N = temp_X.shape[0]
        for ti, pi in self.probabilities.items():
            Ni = self._get_Ni(temp_X, self.score_vector, ti)
            Hi = entropy([pi, 1 - pi], base=2)
            sum += (Ni / N) * Hi
        return sum


class ProbabilisticScoringList(BaseEstimator, ClassifierMixin):
    def __init__(self, entropy_threshold, score_set):
        """ IMPORTANT: Shannon entropy is calculated with respect to base 2
        """
        # self.logger = logging.getLogger(__name__)
        self._estimator_type = "classifier"
        self.calibrator = None
        self.score_set = score_set
        self.score_set = sorted(self.score_set)[::-1]
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
            # try all features and possible scores
            fk, sk = remaining_feature_indices[0], self.score_set[-1]
            current_expected_entropy = np.inf

            if predefined_features is None:
                features_to_consider = remaining_feature_indices
            else:
                features_to_consider = [predefined_features[stage - 1]]

            scores_to_consider = self.score_set if predefined_scores is None else [predefined_scores[stage - 1]]

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
        :param k: Classfier stage to use for prediction
        :return:
        """

        return self.predict_proba(X, k).argmax(axis=1)

    def predict_proba(self, X, k=-1):
        """
        Predicts the probability using the k-th or last classifier
        :param X: Dataset to predict the probabilities for
        :param k: Classfier stage to use for prediction
        :return:
        """

        return self.classifier_at_k[k].predict_proba(X)


def powerset(iterable):
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s) + 1))
