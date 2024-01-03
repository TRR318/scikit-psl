import pytest

from skpsl.metrics import expected_entropy_loss


@pytest.mark.parametrize("y_prob,expected", [([0.5, 0.5], 1), ([0.5, 0], 0.5)])
def test_expected_entropy_loss(y_prob, expected):
    assert expected_entropy_loss(y_prob) == expected


@pytest.mark.parametrize(
    "y_prob,w,expected",
    [
        ([0.5, 0], [0.5, 0.5], 0.5),
        ([0.5, 0], [1, 0], 1),
        ([0.5, 0], [0.2, 0.8], 0.2),
    ],
)
def test_expected_entropy_loss_weighted(y_prob, w, expected):
    assert expected_entropy_loss(y_prob, w) == expected
