[![License](https://img.shields.io/github/license/stheid/scikit-psl)](https://github.com/stheid/scikit-psl/blob/master/LICENSE)
[![Pip](https://img.shields.io/pypi/v/scikit-psl)](https://pypi.org/project/scikit-psl)


# Probabilistic Scoring Lists

# Install
```bash
pip install scikit-psl
```

# Usage

```python
from sklearn.datasets import make_classification
from sklearn.model_selection import ShuffleSplit

from skpsl import ProbabilisticScoringList

# Generating synthetic data with continuous features and a binary target variable
X, y = make_classification(random_state=42)
X = (X > .5).astype(int)

clf = ProbabilisticScoringList([-1, 1, 2])

for train, test in ShuffleSplit(1, test_size=.2, random_state=42).split(X):
    psl = ProbabilisticScoringList([-1, 1, 2])
    psl.fit(X[train], y[train])
    print(f"Brier score: {psl.score(X[test], y[test]):.4f}")
    #>  Brier score: 0.1924  (lower is better)

    df = psl.inspect(5)
    print(df.to_string(index=False, na_rep="-", justify="center", float_format=lambda x: f"{x:.2f}"))    
    #>  Stage  Score  T = -3  T = -2  T = -1  T = 0  T = 1  T = 2  T = 3
    #>   0        -       -       -       -   0.54      -      -      - 
    #>   1     2.00       -       -       -   0.18      -   0.97      - 
    #>   2    -1.00       -       -    0.00   0.28   0.91   1.00      - 
    #>   3    -1.00       -    0.00    0.07   0.86   0.91   1.00      - 
    #>   4     1.00       -    0.00    0.00   0.29   0.92   1.00   1.00 
    #>   5    -1.00    0.00    0.00    0.00   0.40   1.00   1.00   1.00
```
