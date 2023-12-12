from operator import itemgetter

from sklearn.metrics import precision_recall_curve


def precision_at_recall_k_score(
    y_true, y_prob, *, recall_level=0.9, return_threshold=False
):
    # maximum precision for a given recall level
    prec, threshold = max(
        (
            (p, t)
            for p, r, t in zip(*precision_recall_curve(y_true, y_prob))
            if r >= recall_level
        ),
        key=itemgetter(0),
    )
    if return_threshold:
        return prec, threshold
    else:
        return prec


if __name__ == "__main__":
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
