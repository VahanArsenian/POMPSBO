import logging
from abc import ABC, abstractmethod
from pomps.fcm import FunctionalCausalModel, ContextualCausalGraph
from pomps.controllers import GPFunctorFactory, MixedPolicyScope, get_pomps_for, PolicyFCM
from pomis.scm import Domain
from pomps.utils import pareto_optimal, union
import numpy as np
import torch
import typing as tp
from collections import defaultdict
from uuid import uuid4
from tqdm.auto import tqdm
from pathlib import Path
import pickle


logger = logging


class Experiment(ABC):

    def __init__(self):
        pass

    @abstractmethod
    def log_results(self, **kwargs):
        pass


    @abstractmethod
    def step(self):
        pass

    @abstractmethod
    def iterate(self):
        pass


class POMPSExperiment(Experiment):
    def __init__(self, fcm: FunctionalCausalModel, interventional_variables: tp.Set[str],
                 contextual_variables: tp.Set[str], optimization_domain: tp.List[Domain],
                 target: str = "Y", droppable_scopes: tp.List[MixedPolicyScope] = None, is_single_gp=True,
                 n_iter=1500, epsilon=-1):
        super().__init__()
        self.n_iter = n_iter
        self.epsilon = epsilon
        self.is_single_gp = is_single_gp
        self.fcm = fcm
        self.__construct_graphs_under_mps(optimization_domain, interventional_variables, contextual_variables, target)
        self.__drop_undetected(droppable_scopes)
        self.__construct_policies()

        self._active_interventional = union([v.interventional_variables for _, _, v in (self.pomps_active.values())])
        self._active_context = union([v.contextual_variables for _, _, v in (self.pomps_active.values())])

    def __construct_graphs_under_mps(self, optimization_domain, interventional_variables,
                                     contextual_variables, target):
        induced = self.fcm.induced_graph()
        self.ccg = ContextualCausalGraph(edges=induced, interventional_variables={"X1", "X2"},
                                         contextual_variables={"C", "X1"}, target=target)
        assert {s.name for s in optimization_domain}.issuperset(contextual_variables), \
            "Contextual optimization domain is incomplete"
        assert {s.name for s in optimization_domain}.issuperset(interventional_variables), \
            "Interventional optimization domain is incomplete"
        self.factory = GPFunctorFactory(optimization_domain)
        self.graphs_under_mps = get_pomps_for(self.ccg)

    def __drop_undetected(self, droppable_scopes):
        droppable_scopes = {} if droppable_scopes is None else droppable_scopes

        scopes = [scope for _, scope in self.graphs_under_mps]
        for droppable in droppable_scopes:
            try:
                idx: int = scopes.index(droppable)
                self.graphs_under_mps[idx] = None
            except ValueError as _:
                continue
        self.graphs_under_mps = list(filter(None, self.graphs_under_mps))
        self.__results_store = defaultdict(lambda: [])

    def __construct_policies(self):
        self.pomps_active = {}
        for idx, (graph, mps) in enumerate(self.graphs_under_mps):
            if self.is_single_gp:
                policy = PolicyFCM.mps_to_single_gp(mps, self.factory)
            else:
                policy = PolicyFCM.mps_to_gp_policy(mps, self.factory)
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
            self.__results_store['k'].append(v)
        self.__results_store['MPS'].append(str(mps))

    def step(self):
        trial_id = self.__choose_trial()

        fcm_m, policy, mps = self.pomps_active[trial_id]
        if self.is_single_gp:
            print(f"Policy for {policy.variable}, {policy.arguments}")

        print(f"Trial index {trial_id}")
        smp = fcm_m.sample(necessary_context=policy.arguments)
        print(f"Sample is {smp}")

        y = smp[self.ccg.target]
        y = torch.tensor([y])

        if self.is_single_gp:
            policy.functional.observe(-y)
        else:
            for p in policy:
                p.functional.observe(-y)
        self.log_results(smp, mps)

    def iterate(self):
        for i in tqdm(range(self.n_iter)):
            self.step()

    def save_results(self, start, end, prefix="", additional_meta_data: dict = None):
        path = Path(__file__).parent.parent.joinpath("experiment_results")
        if not path.exists():
            path.mkdir()
        path = path.joinpath(f"{prefix}_{uuid4()}.pck")

        meta_data = {"start": start, "end": end, "n_iter": self.n_iter,
                     "epsiolon": self.epsilon, "is_single_gp": self.is_single_gp, }

        if additional_meta_data is not None:
            meta_data.update(additional_meta_data)

        with path.open("wb") as fd:
            pickle.dump({'meta': meta_data, "results": dict(self.__results_store)}, fd)


