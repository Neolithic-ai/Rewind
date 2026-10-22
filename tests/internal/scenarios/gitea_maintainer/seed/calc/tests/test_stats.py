import pytest

from calc import mean, variance


def test_mean():
    assert mean([2, 4, 6]) == 4


def test_variance():
    assert variance([1, 2, 3, 4]) == pytest.approx(1.25)


def test_mean_empty():
    with pytest.raises(ValueError):
        mean([])
