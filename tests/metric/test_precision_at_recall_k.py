from skpsl.metrics import precision_at_recall_k_score
import numpy as np
from pytest import approx
from sklearn.metrics import precision_score

p, t = precision_at_recall_k_score(
    [0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 1, 1, 0, 1, 1, 1],
    np.linspace(0, 1, 16),
    recall_level=0.5,
    return_threshold=True,
)
assert p == approx(0.833, 0.01)
p = precision_score(
    [0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 1, 1, 0, 1, 1, 1], np.linspace(0, 1, 16) >= t
)
assert p == approx(0.833, 0.01)
p = precision_at_recall_k_score(
    [0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 1, 1, 0, 1, 1, 1],
    np.linspace(0, 1, 16),
)
assert p == approx(0.615, 0.01)
