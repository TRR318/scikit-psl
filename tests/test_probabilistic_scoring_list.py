import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline

from skpsl import ProbabilisticScoringList
from skpsl.preprocessing import MinEntropyBinarizer


@pytest.fixture
def numpy_randomness():
    np.random.seed(0)


def test_binary_data():
    X, y = make_classification(random_state=42)
    X = (X > 0.5).astype(int)

    psl = ProbabilisticScoringList({-1, 1, 2})

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    psl.fit(X_train, y_train)
    assert psl.thresholds == [np.nan] * X.shape[1]


def test_inspect():
    # Generating synthetic data with continuous features and a binary target variable
    X, y = make_classification(n_features=5, random_state=42)
    X = (X > 0.5).astype(int)

    psl = ProbabilisticScoringList({-1, 1, 2})

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    psl.fit(X_train, y_train)
    df = psl.inspect(3)
    print(
        df.to_string(
            index=False, na_rep="-", justify="center", float_format=lambda x: f"{x:.2f}"
        )
    )


def test_improvement_for_internalized_binarization():
    X, y = make_classification(n_features=6, n_informative=4, random_state=42)

    clf_ = Pipeline(
        [
            ("binarizer", MinEntropyBinarizer()),
            ("psl", ProbabilisticScoringList({-1, 1, 2})),
        ]
    )
    pipe_score = cross_val_score(clf_, X, y, cv=5).mean()

    clf_ = ProbabilisticScoringList({-1, 1, 2})
    psl_score = cross_val_score(clf_, X, y, cv=5).mean()

    # lower score is better
    # the internalized binarization should perform better
    assert psl_score < pipe_score


def test_gh1_maximally_negative_score_for_first_feature():
    X, y = make_classification(n_features=6, n_informative=4, random_state=42)
    X = (X > 0.5).astype(int)
    y = 1 - X[:, 2]
    psl = ProbabilisticScoringList({-2, -1, 1, 2})
    psl.fit(X, y)
    assert psl.scores[0] == -2


def test_only_negative_classes():
    X, y = make_classification(n_features=6, n_informative=4, random_state=42)
    y.fill(0)
    psl = ProbabilisticScoringList({-2, -1, 1, 2})
    psl.fit(X, y)
    assert not np.isnan(psl.stage_clfs[0].score(X))


def test_sample_weight():
    X, y = make_classification(
        n_samples=10, n_features=6, n_informative=4, random_state=42
    )
    psl = ProbabilisticScoringList({-1, 1})
    weighted = psl.fit(X, y).score(
        X,
        y,
        sample_weight=np.random.exponential(
            X.shape[0],
        ),
    )
    unweighted = psl.fit(X, y).score(X, y)
    assert unweighted != weighted


def test_dataframe():
    X, y = make_classification(
        n_samples=10, n_features=6, n_informative=4, random_state=42
    )
    X = pd.DataFrame(X)
    y = pd.Series(y)

    psl = ProbabilisticScoringList({-1, 1})
    psl.fit(X, y).score(X, y)



def test_predef():
    from sklearn.datasets import make_classification
    from sklearn.model_selection import cross_val_score

    # Generating synthetic data with continuous features and a binary target variable
    X_, y_ = make_classification(random_state=42)

    clf_ = ProbabilisticScoringList({-1, 1, 2}, lookahead=2)
    clf_.fit(X_, y_, predef_features=[3, 2, 1], predef_scores=[2, 2, 1], strict=True)
