from experiments.pomps_experiment import *
from pomis.graphs import *
from npsem.where_do import POMISs
from pomps.hebo_adapted import CustomEI
from pomps.policy_scope import PolicyComponent


class CaBOExperiment(Experiment):
    def _fcm_with_policy(self, mps):
        return PolicyFCM.mps_to_single_gp(mps, self.factory)

    def __init__(self, fcm: FunctionalCausalModel, non_interventional_variables: tp.Set[str],
                 optimization_domain: tp.List[Domain], target: str = "Y",
                 n_iter=1500, epsilon=-1, objetive=OptimizationObjective.maximize, debug=False,
                 auto_log_each_n_iter=50, experiment_name=None):
        super().__init__(fcm, n_iter, epsilon, objetive, debug, auto_log_each_n_iter, experiment_name)

        self.__construct_graphs_under_policy(optimization_domain, non_interventional_variables, target)
        self._construct_policies()

        self._active_interventional = union([v.interventional_variables for _, _, v in (self.policies_active.values())])
        self._active_context = union([v.contextual_variables for _, _, v in (self.policies_active.values())])

    def __construct_graphs_under_policy(self, optimization_domain, non_interventional_variables, target):
        induced = self.fcm.induced_graph()
        nmg = NonManGraph(non_interventional_variables)
        nmg.add_edges_from(induced.edges)
        pomis_s = POMISs(CausalGraph(nmg.projection), target)

        interventional_variables = set(induced.nodes) - non_interventional_variables - set(target)

        self.__contains_empty = set() in pomis_s
        if self.__contains_empty:
            pomis_s: tp.FrozenSet[tp.FrozenSet[str]] = frozenset({p for p in pomis_s if p != set()})

        mps_cmp = self.__convert_pomis_to_mps(pomis_s)
        self.ccg = ContextualCausalGraph(edges=induced, interventional_variables=interventional_variables,
                                         contextual_variables=set(), target=target)
        assert {s.name for s in optimization_domain}.issuperset(interventional_variables), \
            "Interventional optimization domain is incomplete"
        self.graphs_under_policies = [(MPSDAGController.graph_under_mps(mps, self.ccg), mps) for mps in mps_cmp]
        self.factory = GPFunctorFactory(optimization_domain, acq_function=CustomEI)

    @classmethod
    def __convert_pomis_to_mps(cls, pomis_s: tp.FrozenSet[tp.FrozenSet[str]]) -> tp.List[MixedPolicyScope]:
        mps = []
        for pomis in pomis_s:
            components = {PolicyComponent(inter_var, set()) for inter_var in pomis}
            mps.append(
                MixedPolicyScope(components)
            )
        return mps

    def step(self):
        y, policy, smp, mps, trial_id = super().step()
        policy.functional.observe(self._opt_factor*y)

        self.log_results(smp, mps)

    def save_results(self, start, end, prefix="", meta_data: dict = None):
        md = {"start": start, "end": end, "n_iter": self.n_iter,
                     "epsiolon": self.epsilon}
        if meta_data is not None:
            md.update(meta_data)
        super().save_results(start, end, prefix, md)
