from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

from skpsl import ProbabilisticScoringList

if __name__ == '__main__':
    X, y = make_classification(n_informative=10, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=42)

    psl = ProbabilisticScoringList({-1, 1, 2})
    psl.fit(X_train, y_train)
    print(f"Brier score: {psl.score(X_test, y_test):.4f}")

    df = psl.inspect(5)
    print(df.to_string(index=False, na_rep="-", justify="center", float_format=lambda x: f"{x:.2f}"))
