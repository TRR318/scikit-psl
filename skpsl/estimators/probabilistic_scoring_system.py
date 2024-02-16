import logging
from typing import Optional

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.exceptions import NotFittedError
from sklearn.isotonic import IsotonicRegression

from skpsl.metrics import expected_entropy_loss
from skpsl.preprocessing import SigmoidTransformer
from skpsl.preprocessing.beta_transform import BetaTransformer


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
                self.loss_function_ = (
                    lambda true, prob, sample_weight=None: loss_function(true, prob)
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
            is_data_binary = set(np.unique(feature_values).astype(int)) <= {0, 1}
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

    def predict_proba(self, X):
        """
        Predicts the probability for
        """
        if self.calibrator is None:
            raise NotFittedError()
        proba_pos = self.calibrator.transform(
            self._compute_total_scores(
                X, self.features, self.scores_, self.feature_thresholds
            )
        )
        return np.vstack([1 - proba_pos, proba_pos]).T

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
