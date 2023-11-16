import logging
from itertools import permutations, product, combinations_with_replacement
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
from skpsl.preprocessing import SigmoidTransformer


class _ClassifierAtK(BaseEstimator, ClassifierMixin):
    """
    Internal class for the classifier at stage k of the probabilistic scoring list
    """

    def __init__(
        self,
        features: list[int],
        scores: list[int],
        initial_thresholds: list[Optional[float]],
        threshold_optimizer: Optional[callable] = None,
        calibration_method="isotonic",
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
        self.calibration_method = calibration_method

        self.thresholds = list(initial_thresholds)
        self.scores_vec = np.array(scores)
        self.logger = logging.getLogger(__name__)
        self.calibrator = None

    def fit(self, X, y) -> "_ClassifierAtK":
        for i, (f, t) in enumerate(zip(self.features, self.initial_thresholds)):
            feature_values = X[:, f]
            is_data_binary = set(np.unique(feature_values).astype(int)) <= {0, 1}
            if (t is np.nan or t is None) and not is_data_binary:
                self.logger.debug(
                    f"feature {f} is non-binary and threshold not set: calculating threshold..."
                )

                # fit optimal threshold
                self.thresholds[i] = self.threshold_optimizer(
                    lambda t_, _: self._expected_entropy(
                        self._compute_total_scores(
                            X=X,
                            features=self.features[: i + 1],
                            scores=self.scores_vec[: i + 1],
                            thresholds=self.thresholds[:i] + [t_],
                        ),
                        y,
                    ),
                    feature_values,
                )

        total_scores = _ClassifierAtK._compute_total_scores(
            X, self.features, self.scores_vec, self.thresholds
        )

        # compute probabilities
        match self.calibration_method:
            case "isotonic":
                self.calibrator = IsotonicRegression(
                    y_min=0.0, y_max=1.0, increasing=True, out_of_bounds="clip"
                )
            case "sigmoid":
                self.calibrator = SigmoidTransformer()
            case _:
                raise ValueError(
                    f'Calibrationmethod "{self.calibration_method}" does not exist. did you mean "isotonic" or "sigmoid"'
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
            self._compute_total_scores(
                X, self.features, self.scores_vec, self.thresholds
            )
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
        total_scores = self._compute_total_scores(
            X, self.features, self.scores_vec, self.thresholds
        )
        return self._expected_entropy(total_scores, calibrator=self.calibrator)

    @staticmethod
    def _compute_total_scores(X, features, scores: np.ndarray, thresholds):
        if len(features) == 0:
            return np.zeros((X.shape[0], 1))
        data = np.array(X)[:, features]
        thresholds = np.array(thresholds, dtype=float)
        thresholds[np.isnan(thresholds)] = 0.5
        return ((data > thresholds[None, :]) @ scores).reshape(-1, 1)

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
        score_freqs = score_freqs / score_freqs.sum()
        score_probas = np.array(calibrator.transform(total_scores.reshape(-1, 1)))
        entropy_values = entropy([score_probas, 1 - score_probas], base=2)
        return entropy_values @ score_freqs


class ProbabilisticScoringList(BaseEstimator, ClassifierMixin):
    """
    Probabilistic scoring list classifier.
    A probabilistic classifier that greedily creates a PSL selecting one feature at a time
    """

    def __init__(
        self,
        score_set: set,
        loss_cutoff: float = None,
        method="bisect",
        lookahead=1,
        n_jobs=None,
        stage_loss=None,
        cascade_loss=None,
        stage_clf_params=None,
    ):
        """

        :param score_set: Set score values to be considered. Basically feature weights.
        :param loss_cutoff: minimal loss at which to stop fitting further stages. None means fitting the whole stage
        """
        self.score_set = score_set
        self.loss_cutoff = loss_cutoff
        self.method = method
        self.lookahead = lookahead
        self.n_jobs = n_jobs
        self.stage_loss = stage_loss
        self.cascade_loss = cascade_loss
        self.stage_clf_params = stage_clf_params

        self.stage_clf_params_ = (self.stage_clf_params or dict()) | dict(
            threshold_optimizer=create_optimizer(method)
        )
        match stage_loss:
            case None:
                self.stage_loss_ = lambda clf, X, y: clf.score(X)
            case _:
                self.stage_loss_ = lambda clf, X, y: self.stage_loss(
                    y, clf.predict_proba(X)[:, 1]
                )
        match cascade_loss:
            case None:
                self.cascade_loss_ = (
                    lambda losses: sum(losses)
                    + np.minimum(np.array(losses)[1:] - np.array(losses)[:-1], 0).sum()
                )
            case _:
                self.cascade_loss_ = cascade_loss
        self.logger = logging.getLogger(__name__)
        self.score_set_ = np.array(sorted(self.score_set, reverse=True, key=abs))
        self.stage_clfs = []  # type: list[_ClassifierAtK]

    def fit(
        self,
        X,
        y,
        predef_features: Optional[np.ndarray] = None,
        predef_scores: Optional[np.ndarray] = None,
    ) -> "ProbabilisticScoringList":
        """
        Fits a probabilistic scoring list to the given data

        :param X:
        :param y:
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
        losses = [self._fit_and_store_clf_at_k(X, y)]

        while remaining_features and (
            self.loss_cutoff is None or losses[-1] > self.loss_cutoff
        ):
            stage += 1

            # noinspection PyUnresolvedReferences
            features_to_consider = (
                remaining_features
                if predef_features is None
                else [predef_features[stage - 1]]
            )
            scores_to_consider = (
                self.score_set_ if predef_scores is None else [predef_scores[stage - 1]]
            )

            len_ = min(self.lookahead, len(features_to_consider))

            new_cascade_losses, f, s, t = zip(
                *Parallel(n_jobs=self.n_jobs)(
                    delayed(self._optimize)(
                        list(f_seq),
                        list(s_seq),
                        losses,
                        X,
                        y,
                    )
                    for (f_seq, s_seq) in product(
                        permutations(features_to_consider, len_),
                        combinations_with_replacement(scores_to_consider, len_),
                    )
                )
            )

            i = np.argmin(new_cascade_losses)
            remaining_features.remove(f[i])

            losses.append(
                self._fit_and_store_clf_at_k(
                    X,
                    y,
                    self.features + [f[i]],
                    self.scores + [s[i]],
                    self.thresholds + [t[i]],
                )
            )
        return self

    def _fit_and_store_clf_at_k(self, X, y, f=None, s=None, t=None):
        f, s, t = f or [], s or [], t or []
        k_clf = _ClassifierAtK(
            features=f,
            scores=s,
            initial_thresholds=t,
            **self.stage_clf_params_,
        ).fit(X, y)
        self.stage_clfs.append(k_clf)
        return self.stage_loss_(k_clf, X, y)

    def _optimize(
        self,
        feature_extension,
        score_extension,
        cascade_losses,
        X,
        y,
    ):
        # build cascade extension
        new_thresholds = []
        for i in range(1, len(feature_extension) + 1):
            clf = _ClassifierAtK(
                features=self.features + feature_extension[:i],
                scores=self.scores + score_extension[:i],
                initial_thresholds=self.thresholds + new_thresholds + [None],
                **self.stage_clf_params_,
            ).fit(X, y)
            cascade_losses.append(self.stage_loss_(clf, X, y))
            new_thresholds.append(clf.thresholds[-1])

        return (
            self.cascade_loss_(cascade_losses),
            feature_extension[0],
            score_extension[0],
            new_thresholds[0],
        )

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
        if self.stage_clfs is None:
            raise NotFittedError(
                "Please fit the probabilistic scoring classifier before usage."
            )

        return self.stage_clfs[k].predict_proba(X)

    def score(self, X, y, k=None, sample_weight=None):
        """
        Calculates the Brier score of the model
        :param X:
        :param y:
        :param k: Classifier stage to use for prediction
        :param sample_weight: ignored
        :return:
        """
        if self.stage_clfs is None:
            raise NotFittedError()
        if k is None:
            return self.cascade_loss_(
                [
                    brier_score_loss(y, self.predict_proba(X, k=k)[:, 1])
                    for k in range(len(self))
                ]
            )
        return brier_score_loss(y, self.predict_proba(X, k=k)[:, 1])

    def searchspace_analysis(self, X):
        """
        Prints some useful information about the search space

        :param X: only used to derive number of features
        :return: None
        """
        f, s, l = X.shape[1], len(self.score_set), self.lookahead

        # models = calculate number of models at each stage
        print(f"Searchspace size: {(s+1)**f:.2g}")

        # calculate lookahead induced subspace
        effective_searchspace = sum(
            (
                s ** min(l, k)
                * np.array([k_ + 1 for k_ in range(max(k - l, 0), k)]).prod()
                for k in range(f, 0, -1)
            )
        )
        print(f"Models to evaluate (ignoring caching): {effective_searchspace:g}")

    def inspect(self, k=None, feature_names=None) -> pd.DataFrame:
        """
        Returns a dataframe that visualizes the internal model

        :param k: maximum stage to include in the visualization (default: all stages)
        :param feature_names: names of the features.
        :return:
        """
        k = k or len(self) - 1

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

    def __len__(self):
        if self.stage_clfs is None:
            return 0
        return len(self.stage_clfs)

    @property
    def features(self) -> list[int]:
        return self.stage_clfs[-1].features if self.stage_clfs else []

    @property
    def scores(self) -> list[int]:
        return self.stage_clfs[-1].scores if self.stage_clfs else []

    @property
    def thresholds(self) -> list[int]:
        return self.stage_clfs[-1].thresholds if self.stage_clfs else []


if __name__ == "__main__":
    from sklearn.datasets import make_classification
    from sklearn.model_selection import cross_val_score

    # Generating synthetic data with continuous features and a binary target variable
    X_, y_ = make_classification(random_state=42)

    clf_ = ProbabilisticScoringList({-1, 1, 2})
    clf_.searchspace_analysis(X_)
    print("Total Brier score:", cross_val_score(clf_, X_, y_, cv=5, n_jobs=5).mean())
