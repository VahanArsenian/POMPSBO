from pomps.policy_scope import MixedPolicyScope
from pomps.contextual_graphs import ContextualCausalGraph
from networkx import is_directed_acyclic_graph


class MPSDAGController:

    @classmethod
    def graph_under_mps(cls, mps: MixedPolicyScope, graph: ContextualCausalGraph):
        controlled = mps.interventional_variables
        mut_graph = graph.remove_incoming_edges(controlled)
        mut_graph.add_edges_from(mps.pairs)
        return mut_graph, is_directed_acyclic_graph(mut_graph)

