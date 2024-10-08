import logging
from collections import defaultdict
from typing import Optional, Union

import numpy as np
from scipy.stats import beta
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.exceptions import NotFittedError
from sklearn.isotonic import IsotonicRegression

from skpsl.metrics import expected_entropy_loss
from skpsl.helper.calibrators import BetaTransformer, SigmoidTransformer


class ProbabilisticScoringSystem(BaseEstimator, ClassifierMixin):
    """
    Internal class for the classifier at stage k of the probabilistic scoring list
    """

    def __init__(
        self,
        features: list[int],
        scores: list[int],
        initial_feature_thresholds: Optional[list[Optional[float]]] = None,
        threshold_optimizer: Optional[callable] = None,
        calibration_method="isotonic",
        balance_class_weights=False,
        loss_function=None,
    ):
        """
        Regardless of the stage-loss, thresholds are optimized with respect to expected entropy

        :param features: tuple of feature indices. used for selecting data from X
        :param scores: tuple of scores, corresponding to the feature indices
        :param initial_feature_thresholds: tuple of thresholds to binarize the feature values
        """
        self.features = features
        self.scores = scores
        self.initial_feature_thresholds = initial_feature_thresholds
        self.threshold_optimizer = threshold_optimizer
        self.calibration_method = calibration_method
        self.balance_class_weights = balance_class_weights
        self.loss_function = loss_function

        match loss_function:
            case None:
                self.loss_function_ = (
                    lambda _, y_prob, sample_weight=None: expected_entropy_loss(
                        y_prob, sample_weight=sample_weight
                    )
                )
            case _:
                self.loss_function_ = lambda true, prob, sample_weight=None: (
                    loss_function(true, prob)
                    if sample_weight is None
                    else loss_function(true, prob, sample_weight)
                )

        self.classes_ = None
        self.scores_ = np.array(scores)
        self.feature_thresholds = (
            initial_feature_thresholds
            if initial_feature_thresholds is not None
            else [None] * len(features)
        )
        self.logger = logging.getLogger(__name__)
        self.calibrator = None
        self.class_counts_per_score = None
        self.decision_threshold = 0.5

    def fit(self, X, y, sample_weight=None) -> "ProbabilisticScoringSystem":
        X, y = np.array(X), np.array(y)

        self.classes_ = np.unique(y)
        y = np.array(y == self.classes_[1], dtype=int)

        if self.balance_class_weights:
            n = y.size
            n_pos = np.count_nonzero(y == 1)
            self.decision_threshold = n_pos / n

        for i, (f, t) in enumerate(zip(self.features, self.feature_thresholds)):
            feature_values = X[:, f]
            uniq_f_vals = np.unique(feature_values)
            is_data_binary = len(set(uniq_f_vals)) <= 2 and set(uniq_f_vals.astype(float)) <= {0.0, 1.0}
            if (t is np.nan or t is None) and not is_data_binary:
                self.logger.debug(
                    f"feature {f} is non-binary and threshold not set: calculating threshold..."
                )
                if self.threshold_optimizer is None:
                    raise ValueError(
                        "threshold_optimizer mustn't be None, when non-binary features with unset "
                        "thresholds exist."
                    )

                # fit optimal threshold
                # Note: The threshold is optimized with the expected entropy, regardless of the stageloss used in the PSL
                self.feature_thresholds[i] = self.threshold_optimizer(
                    lambda t_, _: self.loss_function_(
                        y,
                        self._create_calibrator().fit_transform(
                            self._compute_total_scores(
                                X,
                                self.features[: i + 1],
                                self.scores_[: i + 1],
                                self.feature_thresholds[:i] + [t_],
                            ),
                            y,
                        ),
                        sample_weight=sample_weight,
                    ),
                    feature_values,
                )

        total_scores = ProbabilisticScoringSystem._compute_total_scores(
            X, self.features, self.scores_, self.feature_thresholds
        )
        self.calibrator = self._create_calibrator().fit(total_scores, y)
        uniq_total_scores, idx = np.unique(total_scores, return_inverse=True)
        self.class_counts_per_score = defaultdict(lambda: {0: 0, 1: 0}) | {
            int(score): {0: 0, 1: 0}
            | {
                c_: count
                for c_, count in zip(*np.unique(y[idx == i], return_counts=True))
            }
            for i, score in enumerate(uniq_total_scores)
        }

        return self

    def _create_calibrator(self):
        if (
            hasattr(self.calibration_method, "fit")
            and hasattr(self.calibration_method, "transform")
            and isinstance(self.calibration_method, TransformerMixin)
        ):
            return self.calibration_method

        # compute probabilities
        match self.calibration_method:
            case "isotonic":
                return IsotonicRegression(
                    y_min=0.0, y_max=1.0, increasing=True, out_of_bounds="clip"
                )
            case "sigmoid":
                return SigmoidTransformer()
            case "beta":
                return BetaTransformer()
            case "beta_reg":
                return BetaTransformer(penalty="l2")
            case _:
                raise ValueError(
                    f'Calibration method "{self.calibration_method}" doesn\'t exist. '
                    'Did you mean "isotonic" or "sigmoid"'
                )

    def predict(self, X):
        if self.calibrator is None:
            raise NotFittedError()
        return self.classes_[
            np.array(self.predict_proba(X)[:, 1] > self.decision_threshold, dtype=int)
        ]

    def predict_proba(self, X, ci: Optional[Union[float, tuple]] = None):
        """
        Predicts the probability for the given datapoint
        :param X:
        :param ci: if given, it will return triples of probabilities (lower, proba, upper) with Clopper-Pearson confidence intervals.
        The confidence interval must be between 0 and 1. It can also be a tuple of confidence intervals, one for the lower bound, one for the higher.
        """
        if self.calibrator is None:
            raise NotFittedError()
        scores = self._compute_total_scores(
            X, self.features, self.scores_, self.feature_thresholds
        )
        sigma, idx = np.unique(scores, return_inverse=True)
        ps = self.calibrator.transform(sigma.reshape(-1, 1))

        if ci is None:
            proba_pos = ps[idx]
            return np.vstack([1 - proba_pos, proba_pos]).T
        if isinstance(ci, float):
            # binomial proportion ci bounds, scaled by bonferroni correction
            al = au = (1 - ci) / len(sigma)
        else:
            assert isinstance(ci, (tuple, list)) and len(ci) == 2
            al, au = (1 - ci[0]) / len(sigma), (1 - ci[1]) / len(sigma)

        # https://en.wikipedia.org/wiki/Binomial_proportion_confidence_interval#Clopper%E2%80%93Pearson_interval
        ls, us = [], []
        for p, i_ in zip(ps, sigma):
            c = self.class_counts_per_score[int(i_)]
            neg, pos = c[0], c[1]

            l, u = beta.ppf([al / 2, 1 - au / 2], [pos, pos + 1], [neg + 1, neg])
            # make sure the bounds are sensible wrt. proba estimate
            ls.append(min(np.nan_to_num(l, nan=0), p))
            us.append(max(np.nan_to_num(u, nan=1), p))

        # make sure the bounds are monotonic in the scoreset
        ls = np.array([max([l] + ls[:i]) for i, l in enumerate(ls)])
        us = np.array([min([u] + us[i:]) for i, u in enumerate(us)])

        return np.vstack([ls[idx], ps[idx], us[idx]]).T

    def score(self, X, y, sample_weight=None):
        """
        Calculates the expected entropy of the fitted model
        :param X:
        :param y:
        :param sample_weight:
        :return:
        """
        if self.calibrator is None:
            raise NotFittedError()
        return self.loss_function_(
            y, self.predict_proba(X)[:, 1], sample_weight=sample_weight
        )

    @staticmethod
    def _compute_total_scores(X, features, scores: np.ndarray, thresholds):
        X = np.array(X)
        if len(features) == 0:
            return np.zeros((X.shape[0], 1))
        data = np.array(X)[:, features]
        thresholds = np.array(thresholds, dtype=float)
        thresholds[np.isnan(thresholds)] = 0.5
        return ((data > thresholds[None, :]) @ scores).reshape(-1, 1)
