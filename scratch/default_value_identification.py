import sys

# sys.path.insert(0, "/dss/dsshome1/04/ra43rid2/msl/scikit-psl")

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate, cross_val_score
from sklearn.metrics import balanced_accuracy_score, make_scorer
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.pipeline import make_pipeline
import numpy as np
from skpsl.preprocessing import MinEntropyBinarizer
from skpsl.estimators import MultinomialScoringList
from matplotlib import pyplot as plt

from sklearn.datasets import load_iris, load_wine

from ConfigSpace import Configuration, ConfigurationSpace
from smac import HyperparameterOptimizationFacade, Scenario
from math import ceil, floor, sqrt
from statistics import fmean

from joblib import Parallel, delayed

if __name__ == "__main__":
    df = pd.read_csv("../data/player_processed.csv")
    X_soccer = df.iloc[:, 1:]
    y_soccer = df.iloc[:, 0]

    X_iris, y_iris = load_iris(return_X_y=True)
    X_wine, y_wine = load_wine(return_X_y=True)

    def evaluate(config: Configuration, seed: int = 0) -> float:

        def evaluate_for_dataset(name, X, y):
            num_features = X.shape[1]
            num_classes = len(np.unique(y))
            num_scores_to_assign = num_features * num_classes
            args = dict(config)
            args["crossover_type"] = str(args["crossover_type"])
            parents_mating_factor = args.pop("parents_mating_factor")
            mutation_num_genes = (
                args.pop("mutation_num_genes_factor") * num_scores_to_assign
            )
            init_pop_factor_minus_mating = args.pop("init_pop_factor_minus_mating")

            ga_params = args
            ga_params["popsize"] = ceil(sqrt(num_scores_to_assign))
            ga_params["parents_mating"] = floor(
                parents_mating_factor * ga_params["popsize"]
            )
            ga_params["init_pop_factor"] = (
                init_pop_factor_minus_mating + parents_mating_factor
            )
            ga_params["mutation_num_genes"] = floor(mutation_num_genes)

            return (
                name,
                -cross_val_score(
                    MultinomialScoringList(
                        score_set=set(range(-3, 4)),
                        ga_params=ga_params,
                        random_state=seed,
                        cascade_loss=fmean,
                    ),
                    X,
                    y,
                    n_jobs=-1,
                ).mean(),
            )

        names = ["soccer", "iris", "wine"]
        data = [(X_soccer, y_soccer), (X_iris, y_iris), (X_wine, y_wine)]
        results = Parallel(n_jobs=-1)(
            delayed(evaluate_for_dataset)(name, X, y)
            for name, (X, y) in zip(names, data)
        )

        return dict(results)

    cs = ConfigurationSpace(
        {
            "crossover_type": ["single_point", "two_points", "uniform", "scattered"],
            "init_pop_factor_minus_mating": (0.1, 5.0),
            "parents_mating_factor": (0.3, 0.7),
            "init_pop_noise": (0.05, 0.5),
            "mutation_num_genes_factor": (0.1, 0.5),
            "maxiter": [100],
        }
    )

    scenario = Scenario(
        cs,
        deterministic=False,
        objectives=["soccer", "iris", "wine"],
        cputime_limit=72000,
        n_trials=250,
    )
    smac = HyperparameterOptimizationFacade(
        scenario,
        evaluate,
        overwrite=False,
        multi_objective_algorithm=HyperparameterOptimizationFacade.get_multi_objective_algorithm(
            scenario, objective_weights=[1, 1, 1]
        ),
    )
    smac.optimize()
