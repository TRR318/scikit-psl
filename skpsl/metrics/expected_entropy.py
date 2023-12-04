from scipy.stats import entropy


def expected_entropy_loss(y_prob):
    return entropy([1 - y_prob, y_prob], base=2).mean()
