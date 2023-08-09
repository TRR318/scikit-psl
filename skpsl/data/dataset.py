import numpy as np
from itertools import permutations

from sklearn.model_selection import ShuffleSplit

from skpsl.probabilistic_scoring_list import _ClassifierAtK
from skpsl import ProbabilisticScoringList


def lookahead_example():
    """
               scores→ [2,1]   [1,2]
    (2, 5, 1, 4, 3)   0.7839  0.7783 l2_psl.scores=[1, 2] l2_psl.stage_clfs[-1].score(X_)=0.7783  score_l1=0.1944 score_l2=0.1926
    """
    psl = ProbabilisticScoringList(s)
    psl.fit(X_, y_)
    if psl.features == [0, 1] and psl.scores == [2, 1]:
        # the psl ordered the features really in the way that feature 0 is the better one
        # now lets test if we can improve the performance by inverting the scores
        score, invscore = [_ClassifierAtK(features=f, scores=s_).fit(X_, y_, ).score(X_) for s_ in permutations(s)]
        if score > invscore:
            score_l1 = psl.score(X_, y_)
            l2_psl = ProbabilisticScoringList(s).fit(X_, y_, l=2)
            score_l2 = l2_psl.score(X_, y_)
            if score_l2 < score_l1:
                print(
                    f"{w}   {score:.4f}  {invscore:.4f} {l2_psl.scores=} {l2_psl.stage_clfs[-1].score(X_)=}  {score_l1=} {score_l2=}")
                print(np.corrcoef(np.hstack([X_, y_.reshape(-1, 1)].T)))


if __name__ == '__main__':
    X = np.array([[1, 0], [0, 1], [1, 1], [1, 0], [0, 1], [1, 1]])
    y = np.array([0, 0, 0, 1, 1, 1])
    f = [0, 1]
    # s = [2, 1]
    s = [2, 1, -1]

    for w in permutations(range(1, X.shape[0] + 1)):
        X_ = np.repeat(X, np.array(w), axis=0)
        y_ = np.repeat(y, np.array(w), axis=0)

        rs = ShuffleSplit(n_splits=5, test_size=0.2, random_state=0)
        for (train_index, test_index) in rs.split(X):
            l1_psl = ProbabilisticScoringList(s).fit(X_[train_index], y_[train_index])
            l1_out = l1_psl.score(X_[test_index], y_[test_index])
            l1_in = l1_psl.score(X_[train_index], y_[train_index])

            l2_psl = ProbabilisticScoringList(s).fit(X_[train_index], y_[train_index], l=2)
            l2_out = l2_psl.score(X_[test_index], y_[test_index])
            l2_in = l2_psl.score(X_[train_index], y_[train_index])
            if l1_in > l2_in:
                print(f"{l1_in=} {l2_in=} {l1_out=} {l2_out=}")
