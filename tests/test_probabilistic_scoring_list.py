import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

from skpsl import ProbabilisticScoringList


def test_binary_data():
    X, y = make_classification(random_state=42)
    X = (X > .5).astype(int)

    psl = ProbabilisticScoringList({-1, 1, 2})

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=42)
    psl.fit(X_train, y_train)
    assert psl.thresholds == [np.nan] * X.shape[1]


def test_inspect():
    # Generating synthetic data with continuous features and a binary target variable
    X, y = make_classification(n_features=5, random_state=42)
    X = (X > .5).astype(int)

    psl = ProbabilisticScoringList({-1, 1, 2})

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=42)
    psl.fit(X_train, y_train)
    df = psl.inspect(3)
    print(df.to_string(index=False, na_rep="-", justify="center", float_format=lambda x: f"{x:.2f}"))
