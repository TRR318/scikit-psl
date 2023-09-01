import numpy as np
from sortedcontainers import SortedSet


def logarithmic_optimizer(func: callable, data: np.array, minimize=True) -> float:
    """
    This algorithm employs a hierarchical logarithmic search to find the global minimum of a parametrized metric.

    Parameter data is used to calculate potential threshold values used for cutting data.
    It is passed to func without changes. The function is assumed to be pseudo convex in x.
    This allows for the hierarchical logarithmic search to find a global optimum efficiently.

    :param data: one-dimensional float array of the feature variable
    :param func: Metric must have the signature (x, data) and return a score to be minimized
    :return: optimal threshold
    """
    values = np.sort(np.unique(data))
    # Adding the extremal values <min and >max might not be necessary, but it is not trivial to proof.
    cuts = np.concatenate([[data.min() - 1], (values[:-1] + values[1:]) / 2, [data.max() + 1]])
    minimal_points = set()
    min_ = np.inf
    thresh = None
    evaluated = SortedSet()
    to_evaluate = {0, cuts.size - 1}

    while to_evaluate:
        # evaluate points
        while to_evaluate:
            k = to_evaluate.pop()
            evaluated.add(k)
            entropy = (-1)**int(not minimize) * func(cuts[k], data)
            if entropy < min_:
                min_ = entropy
                thresh = cuts[k]
                minimal_points = {k}
            elif entropy == min_:
                minimal_points.add(k)

        # calculate new points to evaluate
        for k in minimal_points:
            k_pos = evaluated.index(k)
            candidates = set()
            for offset in [-1, 1]:
                try:
                    candidates.add(k + (evaluated[k_pos + offset] - k) // 2)
                except IndexError:
                    pass
            to_evaluate.update(candidates - evaluated)
    return thresh
