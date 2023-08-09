[![License](https://img.shields.io/github/license/stheid/scikit-psl)](https://github.com/stheid/scikit-psl/blob/master/LICENSE)
[![Pip](https://img.shields.io/pypi/v/scikit-psl)](https://pypi.org/project/scikit-psl)


# Probabilistic Scoring Lists

# Install
```bash
pip install scikit-psl
```

# Usage
```python
from skpsl import ProbabilisticScoringList
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score

# Generating synthetic data with continuous features and a binary target variable
X, y = make_classification(random_state=42)
X = (X > .5).astype(int)

clf = ProbabilisticScoringList([-1, 1, 2])
print(cross_val_score(clf, X, y, cv=5))
```
