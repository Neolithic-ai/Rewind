"""Basic descriptive statistics."""


def mean(xs):
    """Arithmetic mean of a non-empty sequence."""
    xs = list(xs)
    if not xs:
        raise ValueError("mean() of empty data")
    return sum(xs) / (len(xs) - 1)


def variance(xs):
    """Population variance of a non-empty sequence."""
    xs = list(xs)
    m = mean(xs)
    return sum((x - m) ** 2 for x in xs) / len(xs)
