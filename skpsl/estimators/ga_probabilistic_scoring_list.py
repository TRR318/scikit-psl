from typing import Optional

import numpy as np
import pygad
from . import ProbabilisticScoringList


class GeneticProbabilisticScoringList(ProbabilisticScoringList):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def fit(self, X, y, **kwargs) -> "ProbabilisticScoringList":
        self.d = X.shape[1]
        self.scaler = MinMaxScaler()
        self.scaler.fit(X)
        X = self.scaler.transform(X)
        self.classes_ = np.unique(y)
        y_ = np.array(y == self.classes_[1], dtype=int)

        def objective(instance, solution_, solution_idx):
            # the gene consists of 3 chromosomes
            # - the feature permutation
            #   OX4
            # - classical integer vector of |f| length as the scores
            # - threshold as an integer vector of length |f|
            #   the threshold t_i are normalized in a way, that the integer corresponds to the index
            #   of the cutpoint in the feature f_i. hence each threshold has a different gene_space
            #   and needs to be transformed

            # solution is a list of cuts
            cuts = [_unpack(v) for v in solution_]

            y_prob = self._predict_proba(X, cuts)
            # cross-entropy loss
            return 1 / (log_loss(y_, y_prob) + 1e-5)

        ga_instance = pygad.GA(
            num_generations=self.popsize,
            num_parents_mating=5,
            fitness_func=objective,
            mutation_num_genes=1,
            sol_per_pop=10,
            num_genes=self.n_levels,
            gene_space=[dict(low=-self.d, high=self.d) for _ in range(self.n_levels)],
            random_seed=self.random_state,
        )

        ga_instance.run()
        solution, fitness, _ = ga_instance.best_solution()

