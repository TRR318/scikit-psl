import numpy as np
from sklearn.metrics import precision_recall_curve


def constrained_precision(y_true, y_prob, *, recall_constraint=.9, return_thresh=False):
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    thresh = thresholds[precision[recall > recall_constraint].argmax()]
    if thresh.size == 0:
        thresh = thresholds[recall.argmax()]
    thresh = thresh.item()
    prec = precision[np.where(thresholds == thresh)].item()
    if return_thresh:
        return prec, thresh
    return prec