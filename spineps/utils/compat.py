from __future__ import annotations


def zip_strict(*iterables):
    """
    A strict version of zip that raises a ValueError if the input iterables have different lengths.

    Converts each iterable to a list to check lengths. This assumes all iterables are finite.

    Args:
        *iterables: Finite iterables to be zipped together.

    Returns:
        An iterator of tuples, where the i-th tuple contains the i-th element from each iterable.

    Raises:
        ValueError: If the input iterables have different lengths.
    """
    lists = [list(it) for it in iterables]
    lengths = [len(lst) for lst in lists]
    if len(set(lengths)) != 1:
        raise ValueError(f"Length mismatch: {lengths}")
    return zip(*lists)
