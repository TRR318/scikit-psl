import numpy as np
from scipy.stats import entropy
from sklearn.isotonic import IsotonicRegression


def expected_entropy(X, y=None, calibrator=None):
    if calibrator is None:
        if y is None:
            raise AttributeError(
                "_expected_entropy must not be called with both 'calibrator' and 'y' being None"
            )
        calibrator = IsotonicRegression(
            y_min=0.0, y_max=1.0, increasing=True, out_of_bounds="clip"
        )
        calibrator.fit(X, y)
    total_scores, score_freqs = np.unique(X, return_counts=True)
    score_probas = np.array(calibrator.transform(total_scores))
    entropy_values = entropy([score_probas, 1 - score_probas], base=2)
    return np.sum((score_freqs / X.size) * entropy_values)
