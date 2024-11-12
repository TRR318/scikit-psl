import inspect
import logging
from itertools import product, repeat
from operator import itemgetter

import numpy as np
import pandas as pd
from joblib.parallel import delayed, Parallel
from pygad import pygad
from scipy.special import softmax
from scipy.stats import rankdata, truncnorm
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.exceptions import NotFittedError
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, accuracy_score
from sklearn.preprocessing import LabelBinarizer
from tqdm import tqdm

from skpsl.preprocessing import MinEntropyBinarizer

LOGGER = logging.getLogger(__name__)


class MulticlassScoringList(ClassifierMixin,BaseEstimator):

    def __init__(self, score_set, method=None, cascade_loss=None, random_state=None, ga_params=None, **kwargs):
        """
        :param score_set:
        :param cascade_loss: function to aggregate the stage losses
        - None: will give the sum of the whole cascade
        - lambda x:x[-1]: will give the loss of the last stage
        """
        self.score_set = score_set
        self.cascade_loss = cascade_loss
        self.method = method if method is not None else "greedy"
        self.random_state = random_state
        self.ga_params = ga_params

        self.score_set_ = np.array(sorted(score_set, reverse=True, key=abs))
        self.scores = None
        self.f_ranks = None
        self.classes_ = None
        self.binarizer = None
        self.majority_class_idx = None
        self.n_classes = None
        self.n_features = None
        self.ga_instance = None
        self.stage = None
        self.l2 = kwargs.get("l2", 0)
        self.n_jobs = kwargs.get("n_jobs", 1)
        ga_default = dict(maxiter=50, popsize=10, init_pop_factor=1, init_pop_noise=.2, parents_mating=5)
        self.ga_params_ = ga_default | ga_params if ga_params is not None else None
        match cascade_loss:
            case None:
                self.cascade_loss_ = sum
            case _:
                self.cascade_loss_ = cascade_loss

    def __iter__(self):
        self.stage = 0
        while self.stage != len(self) + 1:
            yield self
            self.stage = self.stage + 1
        self.stage = None

    def __getitem__(self, item):
        self.stage = item
        return self

    def fit(self, X, y):
        self.classes_, counts = np.unique(y, return_counts=True)
        self.n_classes = self.classes_.size
        n_instances, self.n_features = X.shape

        self.binarizer = MinEntropyBinarizer().fit(X, y)
        X = self.binarizer.transform(X)

        # Compute cross-entropy loss
        y_trans = LabelBinarizer().fit_transform(y)

        match self.method:
            case "greedy":
                features, scores = [], []

                X = np.hstack((np.ones((n_instances, 1)), X))

                def opt(s, f, logits):
                    logits_ = logits + X[:, [f + 1]] @ np.array([s])
                    return f, list(s), logits_, log_loss(y_trans, softmax(logits_, axis=1)) + self.l2 * (
                            (np.array(s) / max(abs(self.score_set_))) ** 2).mean()

                logits = np.zeros((n_instances, self.n_classes))  # np.repeat([s], n_instances, axis=0)

                # bias term
                res = tqdm(
                    Parallel(n_jobs=1, return_as="generator")(
                        delayed(opt)(s, -1, logits)
                        for (s) in
                        sorted(product(*repeat(self.score_set_, self.n_classes)), key=lambda x: sum(abs(np.array(x))))
                    ),
                    total=len(self.score_set_) ** self.n_classes)
                _, s, logits, _ = min(res, key=itemgetter(3))
                LOGGER.info(f"bias terms: {s}")
                scores.append(s)

                # n features
                remaining_features = set(range(self.n_features))
                while remaining_features:
                    res = tqdm(
                        Parallel(n_jobs=self.n_jobs, return_as="generator")(
                            delayed(opt)(s, f, logits)
                            for (f, s) in
                            product(remaining_features, product(*repeat(self.score_set_, self.n_classes)))
                        ),
                        total=len(remaining_features) * len(self.score_set_) ** self.n_classes)

                    f, s, logits, _ = min(res, key=itemgetter(3))

                    LOGGER.info(f"scores for stage {len(scores)}: {s}")
                    features.append(f)
                    scores.append(s)
                    remaining_features.remove(f)

                # convert orderings to ranks
                self.f_ranks = np.argsort(features)
                # index ordering scores by ranks to get scores in feature permutation (congrent to the f_ranks)
                self.scores = np.array(scores)[[0] + list(self.f_ranks + 1)].T

            case "ga":
                # fit lr as a seed for the genetic search
                lr = LogisticRegression().fit(X, y)

                # FEATURE RANKINGS
                self.f_ranks = np.argsort(lr.coef_.mean(axis=0))

                # CALCULATE SCORES
                # extract logits from LR and rescale and round to the score_set
                # (this destroys the proper scaling for the softmax and hence probability estimates)
                self.scores = self._rescale_round_closest(np.hstack([lr.intercept_.reshape(-1, 1), lr.coef_]))

                # FIT SCALE PARAMETERS
                if self.ga_params_ is not None:
                    # do GA optimization
                    X_trans = np.array(self.binarizer.transform(X))
                    y_trans = LabelBinarizer().fit_transform(y)

                    def mutation(offspring, ga_instance):
                        new_gen = []
                        for i in range(len(offspring)):
                            rng = np.random.default_rng(ga_instance.random_seed)
                            f_ranks, scores = self._unpack(offspring[0], self.n_features)
                            score_set = np.array(sorted(self.score_set_))

                            # replace the rank of one feature
                            f_ranks[rng.integers(0, len(f_ranks))] = rng.random()

                            # select scores indices to replace and convert into score indices
                            flat_indices = rng.choice(scores.size, ga_instance.mutation_num_genes, replace=False)
                            current_indices = np.searchsorted(score_set, scores.flat[flat_indices])

                            # the clip is still necessary, because the documentation does not say if the bounds
                            # are inclusive, and it still seems to crash from time to time
                            a, b = (0 - current_indices), (len(score_set) - 1 - current_indices)
                            new_indices = np.clip(np.round(
                                truncnorm.rvs(a, b, loc=current_indices, scale=1,
                                              size=ga_instance.mutation_num_genes, random_state=rng)).astype(int), 0,
                                                  len(score_set) - 1)

                            # replace values
                            scores.flat[flat_indices] = score_set[new_indices]

                            new_gen.append(self._pack(f_ranks, scores))

                        return np.array(new_gen)

                    def objective(instance, solution_, solution_idx):
                        f_ranks, scores = self._unpack(solution_, self.n_features)

                        quality = []
                        for k in range(1, len(self) + 1):
                            # we can omit stage 0 from the optimization as it is not influenced by the selection of scores
                            mask = f_ranks < k
                            pred = self.classes_[
                                np.argmax(scores[:, 0] + X_trans[:, mask] @ scores[:, 1:][:, mask].T, axis=1)]
                            quality.append(accuracy_score(y, pred))

                        return sum(quality)

                    seed = self._pack(self.f_ranks, self.scores)
                    rng = np.random.default_rng(self.random_state)
                    initial_pop = [seed]
                    for _ in range(int(self.ga_params_.get("init_pop_factor") * self.ga_params_.get("popsize")) - 1):
                        f_ranks, scores = self._unpack(seed, self.n_features)
                        n = int(scores.size * self.ga_params_.get("init_pop_noise"))
                        scores.flat[rng.choice(scores.size, n, replace=False)] = \
                            self.score_set_[rng.integers(0, self.score_set_.size, n)]
                        initial_pop.append(self._pack(rng.permutation(f_ranks), scores))

                    admissible_args = inspect.getfullargspec(pygad.GA)[0]
                    kwargs = {k: v for k, v in self.ga_params_.items() if k in admissible_args}
                    params = dict(fitness_func=objective,
                                  initial_population=initial_pop,
                                  mutation_type=mutation,  # adaptive
                                  num_genes=(self.n_classes + 1) * self.n_features,
                                  gene_space=[dict(low=0, high=1)] * self.n_features +
                                             [sorted(self.score_set)] * (self.n_features + 1) * self.n_classes,
                                  num_generations=self.ga_params_.get("maxiter"),
                                  num_parents_mating=self.ga_params_.get("parents_mating"),
                                  sol_per_pop=self.ga_params_.get("popsize"),
                                  random_seed=self.random_state,
                                  )
                    self.ga_instance = pygad.GA(**(params | kwargs))
                    self.ga_instance.run()
                    solution, fitness, _ = self.ga_instance.best_solution()
                    self.f_ranks, self.scores = self._unpack(solution, self.n_features)

        return self

    @staticmethod
    def _unpack(solution, m):
        solution = np.array(solution)
        return (rankdata(solution[:m], method="ordinal") - 1).astype(int), solution[m:].reshape(-1, m + 1).astype(int)

    @staticmethod
    def _pack(f_ranks, scores):
        return list(f_ranks / len(f_ranks)) + list(scores.flatten())

    def _rescale_round_closest(self, values):
        # self.score_set is sorted differently (by absolute value) so we need to resort it.
        sorted_scoreset = np.array(sorted(self.score_set_))

        scale = max(values.min() / sorted_scoreset.min(), values.max() / sorted_scoreset.max())
        values = values / scale

        # Find the insertion points for each value in values
        pos = np.searchsorted(sorted_scoreset, values)

        # Clip positions to handle edge cases where pos is out of bounds
        pos = np.clip(pos, 1, len(sorted_scoreset) - 1)

        # Get the closest values before and after each position
        before = sorted_scoreset[pos - 1]
        after = sorted_scoreset[pos]

        # Choose the closer value by comparing absolute differences
        return np.where(np.abs(values - before) <= np.abs(values - after), before, after)

    def __len__(self):
        return len(self.f_ranks)

    def predict(self, X):
        # get random argmax
        arr = self._logit(X, self.stage)
        is_max = arr == arr.max(axis=1, keepdims=True)  # Boolean mask of max locations
        rand_vals = np.random.rand(*arr.shape) * is_max  # Assign random values only to max locations
        return self.classes_[rand_vals.argmax(axis=1)]

    def _logit(self, X, k=None):
        if self.scores is None:
            raise NotFittedError()
        X_bin = np.array(self.binarizer.transform(X))
        if k is None or k == -1:
            k = len(self)
        mask = self.f_ranks < k
        return self.scores[:, 0] + X_bin[:, mask] @ self.scores[:, 1:][:, mask].T

    def predict_proba(self, X):
        k = self.stage if self.stage is not None else -1
        return softmax(self._logit(X, k), axis=1)

    def score(self, X, y, sample_weight=None):
        k = self.stage
        if k is not None:
            return accuracy_score(y, self.predict(X))
        loss = self.cascade_loss_([stage.score(X, y) for stage in self])
        self.stage = None
        return loss

    def inspect(self, feature_names=None, class_names=None):
        # when using the f_ranks for indexing they have to be converted into orderings
        f_ordering = np.argsort(self.f_ranks)

        sections = []
        if feature_names is not None:
            feature_names = np.array(feature_names)
            sections.append(pd.DataFrame([""] + feature_names[f_ordering].tolist(), columns=["Feature"]))
        else:
            sections.append(
                pd.DataFrame([-1] + np.array(list(range(self.n_features)))[f_ordering].tolist(), columns=["Feature"]))
        sections.append(
            pd.DataFrame([np.nan] + self.binarizer.threshs[f_ordering].T.tolist(), columns=["Thresholds (>)"]))
        sections.append(pd.DataFrame(self.scores[:, [0] + (f_ordering + 1).tolist()].T,
                                     columns=(class_names if class_names is not None else self.classes_)))
        return pd.concat(sections, axis=1)

    @property
    def features(self):
        return np.where(self.f_ranks < self.stage)[0]

if __name__ == '__main__':
    from sklearn.datasets import load_iris

    data = load_iris()
    X, y = data.data, data.target
    clf = MulticlassScoringList(score_set=set(range(-3, 4)),  # cascade_loss=lambda x: x[-1]
                                method="greedy").fit(X, y)
    print(clf.inspect(data.feature_names, data.target_names))
    clf.predict(X)