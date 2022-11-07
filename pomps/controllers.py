from pomps.policy_scope import MixedPolicyScope
from pomps.contextual_graphs import ContextualCausalGraph
from networkx import is_directed_acyclic_graph, descendants


class MPSDAGController:

    @classmethod
    def graph_under_mps(cls, mps: MixedPolicyScope, graph: ContextualCausalGraph):
        controlled = mps.interventional_variables
        mut_graph = graph.remove_incoming_edges(controlled)
        mut_graph.add_edges_from(mps.pairs)
        return mut_graph, is_directed_acyclic_graph(mut_graph)

    @classmethod
    def simplify(cls, graph: ContextualCausalGraph):
        new_interventional = []
        for inter in graph.interventional_variables:
            related = descendants(graph, inter) & (graph.contextual_variables | {graph.target})
            if len(related) != 0:
                new_interventional.append(inter)
        new_interventional = set(new_interventional)
        graph_related_reduced = graph.remove_outgoing_edges(new_interventional)
        redundant = descendants(graph_related_reduced, graph_related_reduced.target) & new_interventional
        z_redundant = descendants(graph.remove_incoming_edges(redundant),  graph.target)
        new_context = graph.contextual_variables - redundant - z_redundant

        return ContextualCausalGraph(edges=graph.subgraph(set(graph.nodes) - z_redundant),
                                     interventional_variables=new_interventional,
                                     contextual_variables=new_context)
