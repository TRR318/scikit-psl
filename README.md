[![License](https://img.shields.io/github/license/trr318/scikit-psl)](https://github.com/trr318/scikit-psl/blob/master/LICENSE)
[![Pip](https://img.shields.io/pypi/v/scikit-psl)](https://pypi.org/project/scikit-psl)
[![Paper](https://img.shields.io/badge/doi-10.1007%2F978--3--031--45275--8__13-green)](https://doi.org/10.1007/978-3-031-45275-8_13)


# Probabilistic Scoring Lists

Probabilistic scoring lists are incremental models that evaluate one feature of the dataset at a time.
PSLs can be seen as a extension to *scoring systems* in two ways:
- they can be evaluated at any stage allowing to trade of model complexity and prediction speed.
- they provide probablistic predictions instead of deterministic decisions for each possible score.

Scoring systems are used as decision support systems for human experts e.g. in medical or judical decision making.

This implementation adheres to the [sklearn-api](https://scikit-learn.org/stable/glossary.html#glossary-estimator-types).

# Install
```bash
pip install scikit-psl
```

# Usage

For examples have a look at the `examples` folder, but here is a simple example


```python
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

from skpsl import ProbabilisticScoringList

# Generating synthetic data with continuous features and a binary target variable
X, y = make_classification(n_informative=10, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=42)

psl = ProbabilisticScoringList({-1, 1, 2})
psl.fit(X_train, y_train)
print(f"Brier score: {psl.score(X_test, y_test, -1):.4f}")
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
```
