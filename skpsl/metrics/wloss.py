from sklearn.metrics import confusion_matrix


def weighted_loss(y_true, y_prob, m=10, *, sample_weight=None):
    tn, fp, fn, tp = confusion_matrix(
        y_true, 1 - y_prob < m * y_prob, sample_weight=sample_weight, normalize="all"
    ).ravel()
    return fp + m * fn
