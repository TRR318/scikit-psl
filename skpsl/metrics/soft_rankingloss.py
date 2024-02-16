import numpy as np


def bipartite_soft_label_ranking_loss(y_true, y_score):
    y_true, y_score = np.array(y_true), np.array(y_score)
    assert y_true.shape[1] == y_score.shape[1] == 2
    t = np.sign(y_true[:, 0] - y_true[:, 1])
    p = np.sign(y_score[:, 0] - y_score[:, 1])

    # assume that labels contain no ties
    assert np.count_nonzero(t) == t.size

    tie_mask = p == 0

    # ties incure a loss of .5 and false orderings a loss of 1
    return (0.5 * p[tie_mask].size + np.count_nonzero(p[~tie_mask] != t[~tie_mask])) / y_true.shape[0]


def soft_ranking_loss(y_true, y_score):
    # bisect ytrue into pos and neg
    pos_idx = np.where(y_true == 1)[0]
    neg_idx = np.where(y_true != 1)[0]

    # create all pairs of (pos,neg) (as indices)
    tuples = np.array(np.meshgrid(pos_idx, neg_idx)).T.reshape(-1, 2)

    return bipartite_soft_label_ranking_loss(y_true[tuples], y_score[tuples])
