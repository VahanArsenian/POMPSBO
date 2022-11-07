import networkx as nx
import typing as tp
from pomps.utils import union


class ContextualCausalGraph(nx.DiGraph):

    def __init__(self, interventional_variables: tp.Set[str] = None,
                 contextual_variables: tp.Set[str] = None, edges=None,
                 uc_variables: tp.Union[tp.Set[str], str] = None, target='Y'):
        self.interventional_variables = interventional_variables
        self.contextual_variables = contextual_variables
        self.target = target
        super().__init__(edges)
        if uc_variables is None:
            uc_variables = set()
        if uc_variables == 'auto':
            uc_variables = {node for node in self.nodes if node.startswith("U")}
        self.uc_variables = uc_variables
        self.validate_ucs()

    def validate_ucs(self):
        removable = []
        for uc in self.uc_variables:
            confounded = self.children(uc)
            if len(confounded) == 1:
                removable.append(uc)
            if len(confounded) > 2:
                raise ValueError("Single UC confounds more than two variables consider projecting to multiple UCs")
        self.shrink_uc_variables(set(removable))

    def shrink_uc_variables(self, uc_variables: tp.Set[str]):
        assert uc_variables.issubset(self.uc_variables)
        self.remove_nodes_from(self.uc_variables - uc_variables)
        self.uc_variables = self.uc_variables - uc_variables

    def parents(self, target: tp.Union[str, tp.Set[str]], include_uc=False):
        if isinstance(target, str):
            target = {target}
        if include_uc:
            return union([set(self.predecessors(t)) for t in target])
        else:
            return union([set(self.predecessors(t)) for t in target]) - self.uc_variables

    def children(self, target: tp.Union[str, tp.Set[str]]):
        if isinstance(target, str):
            target = {target}
        return union([set(self.successors(t)) for t in target])

    def remove_incoming_edges(self, target: tp.Union[str, tp.Set[str]]) -> "ContextualCausalGraph":
        if isinstance(target, str):
            target = {target}
        copy = ContextualCausalGraph(edges=self, uc_variables=self.uc_variables,
                                     interventional_variables=self.interventional_variables,
                                     contextual_variables=self.contextual_variables)
        for t in target:
            parents = copy.parents(t, include_uc=True)
            edges = [(p, t) for p in parents]
            copy.remove_edges_from(edges)
        copy.validate_ucs()
        return copy

    def remove_outgoing_edges(self, target: tp.Union[str, tp.Set[str]]) -> "ContextualCausalGraph":
        if isinstance(target, str):
            target = {target}
        copy = ContextualCausalGraph(edges=self, uc_variables=self.uc_variables,
                                     interventional_variables=self.interventional_variables,
                                     contextual_variables=self.contextual_variables)
        for t in target:
            children = copy.children(t)
            edges = [(t, c) for c in children]
            copy.remove_edges_from(edges)
        return copy

    def topological_order(self, include_uc=False):
        if include_uc:
            raise NotImplementedError
        return [n for n in nx.topological_sort(self) if n not in self.uc_variables]
