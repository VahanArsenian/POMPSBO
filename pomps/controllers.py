from pomps.policy_scope import MixedPolicyScope, PolicyComponent
from pomps.contextual_graphs import ContextualCausalGraph
from networkx import is_directed_acyclic_graph, descendants, ancestors, d_separated
from pomps.fcm import Functor, FunctionalCausalModel
import typing as tp
from pomps.gp_fcm import GPFunctorFactory


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
        z_redundant = descendants(graph.remove_incoming_edges(redundant), graph.target)
        new_context = graph.contextual_variables - redundant - z_redundant

        return ContextualCausalGraph(edges=graph.subgraph(set(graph.nodes) - z_redundant),
                                     interventional_variables=new_interventional,
                                     contextual_variables=new_context, uc_variables=graph.uc_variables,
                                     target=graph.target)


class PolicyFCM:

    @classmethod
    def fcm_with_policy(cls, fcm: FunctionalCausalModel, policy: tp.Set[Functor]) -> FunctionalCausalModel:
        exog_sampler = fcm.prob_over_exogenous
        base_functors = fcm.functors
        for p in policy:
            base_functors[p.variable] = p
        return FunctionalCausalModel(base_functors.values(), exog_sampler)

    @classmethod
    def mps_to_gp_policy(cls, mps: MixedPolicyScope, factory: GPFunctorFactory):
        return {factory.construct(pc.target, pc.context)
                for pc in mps.components.values()}


class MPSReductor:

    @classmethod
    def action_relevance_check(cls, mps: MixedPolicyScope,
                               mutilated_graph: ContextualCausalGraph):
        x_s = mps.interventional_variables
        return x_s.issubset(ancestors(mutilated_graph, mutilated_graph.target))

    @classmethod
    def context_relevance_check(cls, mps: MixedPolicyScope,
                                mutilated_graph: ContextualCausalGraph):
        for component in mps.components.values():
            h_x = mutilated_graph.subgraph(set(mutilated_graph.nodes) - {component.target})
            for context_var in component.context:
                conditional = component.context - {context_var}
                is_relevant = not d_separated(h_x, {context_var}, {mutilated_graph.target},
                                              mps.implied(conditional) & set(h_x.nodes))
                if not is_relevant:
                    return False
        return True

    @classmethod
    def sufficiently_not_pomp(cls, mps: MixedPolicyScope,
                              mutilated_graph: ContextualCausalGraph):
        c_star = mutilated_graph.contextual_variables
        for component in mps.components.values():
            potential_addition = c_star - component.context - \
                                 {component.target} - descendants(mutilated_graph, component.target)
            h_x = mutilated_graph.subgraph(set(mutilated_graph.nodes) - {component.target})
            for c in potential_addition:
                improves = not d_separated(h_x, {c}, {mutilated_graph.target},
                                       set(h_x.nodes) & mps.implied(component.context))
                if improves:
                    return True
        return False
