import pytest

from skpsl.metrics.soft_rankingloss import bipartite_soft_label_ranking_loss


@pytest.mark.parametrize(
    "y_true,y_score,expected",
    [
        ([[0, 1]], [[0.5, 0.5]], 0.5),
        ([[0, 0.5]], [[0, 0.1]], 0),
        ([[0, 1]], [[0.2, 0]], 1),
    ],
)
def test_expected_entropy_loss_weighted(y_true, y_score, expected):
    assert bipartite_soft_label_ranking_loss(y_true, y_score) == expected
