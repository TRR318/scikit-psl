from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

from skpsl import ProbabilisticScoringList

if __name__ == '__main__':
    X, y = make_classification(n_informative=10, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=42)

    psl = ProbabilisticScoringList({-1, 1, 2})
    psl.fit(X_train, y_train)
    print(f"Brier score: {psl.score(X_test, y_test):.4f}")
    """
    Brier score: 0.2438  (lower is better)
    """

    df = psl.inspect(5)
    print(df.to_string(index=False, na_rep="-", justify="center", float_format=lambda x: f"{x:.2f}"))
    """
     Stage Threshold  Score  T = -2  T = -1  T = 0  T = 1  T = 2  T = 3  T = 4  T = 5
      0            -     -       -       -   0.51      -      -      -      -      - 
      1     >-2.4245  2.00       -       -   0.00      -   0.63      -      -      - 
      2     >-0.9625 -1.00       -    0.00   0.00   0.48   1.00      -      -      - 
      3      >0.4368 -1.00    0.00    0.00   0.12   0.79   1.00      -      -      - 
      4     >-0.9133  1.00    0.00    0.00   0.12   0.12   0.93   1.00      -      - 
      5      >2.4648  2.00    0.00    0.00   0.07   0.07   0.92   1.00   1.00   1.00 
    """
