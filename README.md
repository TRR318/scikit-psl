[![License](https://img.shields.io/github/license/stheid/sklearn-psl)](https://github.com/stheid/sklearn-psl/blob/master/LICENSE)
[![Pip](https://img.shields.io/pypi/v/:sklearn-psl)](https://pypi.org/project/sklearn-psl)

# Probabilistic Scoring Lists

# Install
```bash
pip install psl
```

# Usage
```python
from psl import ProbabilisticScoringList
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score

# Generating synthetic data with continuous features and a binary target variable
X, y = make_classification(random_state=42)
X = (X > .5).astype(int)

clf = ProbabilisticScoringList([-1, 1, 2])
print(cross_val_score(clf, X, y, cv=5))
```