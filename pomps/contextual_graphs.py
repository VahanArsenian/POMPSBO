import networkx as nx
import typing as tp
from pomps.utils import union


class ContextualCausalGraph(nx.DiGraph):

    def __init__(self, edges=None, uc_variables: tp.Union[tp.Set[str], str] = None):
        # interventional_variables: tp.Set[str], contextual_variables: tp.Set[str],
        #          latent_variables: tp.Set[str],
        # self.interventional_variables = interventional_variables
        # self.contextual_variables = contextual_variables
        # self.latent_variables = latent_variables
        # assert len(self.latent_variables & (self.interventional_variables | self.contextual_variables)) == 0, \
        #     "Latent variable can not be contextual or interventional"
        super().__init__(edges)
        if uc_variables is None:
            uc_variables = set()
        if uc_variables == 'auto':
            uc_variables = {node for node in self.nodes if node.startswith("U")}
        self.uc_variables = uc_variables

    def shrink_uc_variables(self, uc_variables: tp.Set[str]):
        assert uc_variables.issubset(self.uc_variables)
        self.remove_nodes_from(self.uc_variables - uc_variables)
        self.uc_variables = self.uc_variables - uc_variables

    def parents(self, target: tp.Union[str, tp.Set[str]]):
        if isinstance(target, str):
            target = {target}
        return union([set(self.predecessors(t)) for t in target])

    def remove_incoming_edges(self, target: tp.Union[str, tp.Set[str]]) -> "ContextualCausalGraph":
        if isinstance(target, str):
            target = {target}
        copy = ContextualCausalGraph(self.edges, self.uc_variables)
        uc_variables = self.uc_variables
        for t in target:
            parents = copy.parents(t)
            uc_variables = uc_variables - parents
            edges = [(p, t) for p in parents]
            copy.remove_edges_from(edges)
        copy.shrink_uc_variables(uc_variables)
        return copy

    def topological_order(self, include_uc=False):
        if include_uc:
            raise NotImplementedError
        return [n for n in nx.topological_sort(self) if n not in self.uc_variables]
