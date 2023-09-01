from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from skpsl import MinEntropyBinarizer, ProbabilisticScoringList

if __name__ == '__main__':
    X, y = make_classification(n_informative=10, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=42)

    # The MinEntropyBinarizer calculates an optimal threshold beforehand.
    # It will calculate a threshold for binarizing the continuous data
    # The optimization criterion is an entropy based impurity measure, similar to the ones used for decision trees.
    # The PSL than operates on the binary data
    pipe = Pipeline([("binarizer", MinEntropyBinarizer()),
                     ("psl", ProbabilisticScoringList({-1, 1, 2}))])
    pipe.fit(X_train, y_train)
    print(f"Brier score: {pipe.score(X_test, y_test):.4f}")
    # >  Brier score: 0.2184  (lower is better)

    df = pipe["binarizer"].inspect()
    print(df.to_string(index=False, na_rep="-", justify="center", float_format=lambda x: f"{x:.2f}"))

    df = pipe["psl"].inspect(5)
    print(df.to_string(index=False, na_rep="-", justify="center", float_format=lambda x: f"{x:.2f}"))
