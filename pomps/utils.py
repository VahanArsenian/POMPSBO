import functools
import itertools
import typing as tp
from scipy.spatial.distance import cdist
import numpy as np


def union(sets: tp.Iterable[tp.Set]) -> tp.Set:
    return functools.reduce(lambda x, y: x | y, sets, set())


def all_combs(base, start=0):
    return list(itertools.chain(*[itertools.combinations(base, i) for i in range(start, len(base) + 1)]))


def inter_cont_pair_gen(interventional_set, contextual_set):
    inter_segment = []
    context_space = all_combs(contextual_set)
    for inter in interventional_set:
        inter_segment += [[(inter, i) for i in context_space]]

    iam = all_combs(inter_segment)
    return itertools.chain(*[(itertools.product(*iam[i])) for i in range(len(iam))], [])


def not_dominated(a, b):
    return (np.asarray(a) >= b).any()


def pareto_optimal(x):
    x = cdist(x, x, metric=not_dominated).astype(np.bool)
    non_dominated = np.where(x.all(axis=1))[0]
    return non_dominated

