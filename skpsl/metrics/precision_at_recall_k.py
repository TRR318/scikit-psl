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