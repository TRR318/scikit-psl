from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

from skpsl import ProbabilisticScoringList


def test_score():
    X, y = make_classification(random_state=42)
    X = (X > .5).astype(int)

    psl = ProbabilisticScoringList({-1, 1, 2})

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=42)
    psl.fit(X_train, y_train)
    assert psl.score(X_test, y_test) < 1
