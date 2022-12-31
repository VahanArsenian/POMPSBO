from experiments.pomps_experiment import *
from pomis.graphs import *
from npsem.where_do import POMISs
from pomps.policy_scope import PolicyComponent


class CaBOExperiment(Experiment):
    def __init__(self, fcm: FunctionalCausalModel, non_interventional_variables: tp.Set[str],
                 optimization_domain: tp.List[Domain], target: str = "Y",
                 n_iter=1500, epsilon=-1, objetive=OptimizationObjective.maximize):
        super().__init__()
        self.__opt_factor = objetive.coefficient()
        self.n_iter = n_iter
        self.epsilon = epsilon
        self.fcm = fcm
        self.__contains_empty = False
        self.__construct_graphs_under_pomis(optimization_domain, non_interventional_variables, target)
        self.__construct_policies()

        self._active_interventional = union([v.interventional_variables for _, _, v in (self.pomps_active.values())])
        self._active_context = union([v.contextual_variables for _, _, v in (self.pomps_active.values())])

    def __construct_graphs_under_pomis(self, optimization_domain, non_interventional_variables, target):
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
        self.graphs_under_mps = [(MPSDAGController.graph_under_mps(mps, self.ccg), mps) for mps in mps_cmp]
        self.factory = GPFunctorFactory(optimization_domain)
        self.__results_store = defaultdict(lambda: [])

    @classmethod
    def __convert_pomis_to_mps(cls, pomis_s: tp.FrozenSet[tp.FrozenSet[str]]) -> tp.List[MixedPolicyScope]:
        mps = []
        for pomis in pomis_s:
            components = {PolicyComponent(inter_var, set()) for inter_var in pomis}
            mps.append(
                MixedPolicyScope(components)
            )
        return mps

    def __construct_policies(self):
        self.pomps_active = {}
        for idx, (graph, mps) in enumerate(self.graphs_under_mps):
            policy = PolicyFCM.mps_to_single_gp(mps, self.factory)
            fcm_m = PolicyFCM.fcm_with_policy(self.fcm, policy)
            self.pomps_active[idx] = (fcm_m, list(policy)[0], mps)

    def __choose_trial(self):
        if np.random.uniform(0, 1) < self.epsilon:
            trial_id = np.random.choice(list(self.pomps_active.keys()))
            print(f"Choosing randomly {trial_id}")
            return trial_id
        trial_index = None
        try:
            trial_index = [idx for idx, (f, p, _) in (self.pomps_active.items()) if p.acq_vals is None][0]
            print(f"None detected in acquisition function. Choosing {trial_index}")
        except IndexError as _:
            fold = np.row_stack([p.acq_vals for f, p, _ in self.pomps_active.values()])
            optimal = pareto_optimal(fold)
            print(f"Optimal indexes {optimal}")
            trial_index = np.random.choice(optimal)
        return trial_index

    def log_results(self, sample: dict, mps: MixedPolicyScope):
        for k, v in sample.items():
            if type(v) is torch.Tensor:
                v = v.item()
            self.__results_store[k].append(v)
        self.__results_store['MPS'].append(str(mps))

    def step(self):
        trial_id = self.__choose_trial()

        fcm_m, policy, mps = self.pomps_active[trial_id]
        print(f"Policy for {policy.variable}, {policy.arguments}")

        print(f"Trial index {trial_id}")
        smp = fcm_m.sample(necessary_context=policy.arguments)
        print(f"Sample is {smp}")

        y = smp[self.ccg.target]
        y = torch.tensor([y])

        policy.functional.observe(self.__opt_factor*y)

        self.log_results(smp, mps)

    def iterate(self):
        for i in tqdm(range(self.n_iter)):
            self.step()

    def save_results(self, start, end, prefix="", additional_meta_data: dict = None):
        path = Path(__file__).parent.parent.joinpath("experiment_results")
        if not path.exists():
            path.mkdir()

        path = path.joinpath(prefix)
        if not path.exists():
            path.mkdir()

        path = path.joinpath(f"{prefix}_{uuid4()}.pck")

        meta_data = {"start": start, "end": end, "n_iter": self.n_iter,
                     "epsiolon": self.epsilon, "is_single_gp": self.is_single_gp, }

        if additional_meta_data is not None:
            meta_data.update(additional_meta_data)

        with path.open("wb") as fd:
            pickle.dump({'meta': meta_data, "results": dict(self.__results_store)}, fd)


