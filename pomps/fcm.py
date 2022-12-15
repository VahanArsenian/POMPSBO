import typing as tp
from copy import copy

import deprecated

from pomps.utils import union
from itertools import combinations
from pomps.contextual_graphs import ContextualCausalGraph
from networkx import is_directed_acyclic_graph


class Functor:

    def __init__(self, functional: tp.Callable, variable: str):
        self.variable = variable
        self.functional = functional
        self.arguments = set(self.functional.__code__.co_varnames)

    def __call__(self, payload: tp.Dict[str, tp.Any]):
        assert set(payload.keys()).issuperset(self.arguments), "Signature mismatch"
        active_payload = {k: v for k, v in payload.items() if k in self.arguments}
        return self.functional(**active_payload)

    def __hash__(self):
        return hash(self.variable)


class FunctionalCausalModel:

    def __init__(self, functors: tp.Set[Functor],
                 sampler_over_exogenous: tp.Callable[..., tp.Dict[str, tp.Any]]):
        self.functors = {f.variable: f for f in functors}
        self.endogenous = set(self.functors.keys())
        self.exogenous = union([f.arguments for f in self.functors.values()]) - self.endogenous
        self.prob_over_exogenous = sampler_over_exogenous

        assert set(sampler_over_exogenous().keys()) == self.exogenous, \
            f"Invalid probability measure over exogenous: " \
            f"vars are: {self.exogenous}, sampler gives: {set(sampler_over_exogenous().keys())}"

    def induced_graph(self):
        # TODO: Cache the induced graph
        puc_counter = {ex: [] for ex in self.exogenous}
        edges = []
        uc_s = []
        for f in self.functors.values():
            for arg in f.arguments:
                if arg in self.exogenous:
                    puc_counter[arg].append((arg, f.variable))
                else:
                    edges.append((arg, f.variable))
        bi_dir_edges = set()
        for exog, ed in puc_counter.items():
            if len(ed) > 1:
                c_component_endog = [edge[1] for edge in ed]
                bi_dir_edges |= set(combinations(c_component_endog, 2))
        for idx, bi_dir_edge in enumerate(bi_dir_edges, 1):
            projected_uc = f"U{idx}"
            uc_s += [projected_uc]
            edges += [(projected_uc, bi_dir_edge[0]), (projected_uc, bi_dir_edge[1])]
        return ContextualCausalGraph(edges=edges, uc_variables=set(uc_s))

    def is_acyclic(self):
        return is_directed_acyclic_graph(self.induced_graph())

    def sample(self, necessary_context: tp.Set[str] = None, return_exog=False):
        sorted_nodes = self.induced_graph().topological_order(necessary_context)
        # TODO: MB cache the order
        exogenous = self.prob_over_exogenous()
        observed = {}
        for n in sorted_nodes:
            if n in observed:
                continue
            payload = copy(exogenous)
            payload.update(observed)
            observed[n] = self.functors[n](payload)
        if return_exog:
            observed.update(exogenous)
        return observed
