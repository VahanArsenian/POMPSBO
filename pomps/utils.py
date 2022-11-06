import functools
import typing as tp


def union(sets: tp.Iterable[tp.Set]) -> tp.Set:
    return functools.reduce(lambda x, y: x | y, sets)
