import numpy as np
from scipy.stats import entropy


def expected_entropy_loss(y_prob, sample_weight=None):
    y_prob = np.array(y_prob)
    return np.average(entropy([1 - y_prob, y_prob], base=2), weights=sample_weight)
