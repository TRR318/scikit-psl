from sklearn.metrics import precision_recall_curve


def precision_at_recall_k_score(y_true, y_prob, *, recall_level=0.9):
    prec = max(
        p
        for p, r, _ in zip(*precision_recall_curve(y_true, y_prob))
        if r >= recall_level
    )
    return prec


if __name__ == "__main__":
    print(precision_at_recall_k_score([1, 1, 1, 0, 0, 0, 0], [0, 0, 0, 1, 1, 1, 1]))
