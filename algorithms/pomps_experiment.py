import datetime
import enum
import logging
from abc import ABC, abstractmethod
from pomps.fcm import FunctionalCausalModel, ContextualCausalGraph
from pomps.controllers import GPFunctorFactory, MixedPolicyScope, get_mps_for, PolicyFCM, \
    MPSDAGController, PolicyComponent
from pomps.hebo_adapted import ReducedMACE, CustomEI
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
from pomps.ucb import UCB

# logger = logging.getLogger("pomps_logger")


class OptimizationObjective(enum.Enum):

    maximize = "maximize"
    minimize = "minimize"

    def coefficient(self):
        if self == OptimizationObjective.minimize:
            return 1
        elif self == OptimizationObjective.maximize:
            return -1
        else:
            raise NotImplementedError


class Experiment(ABC):

    def __init__(self, fcm: FunctionalCausalModel, n_iter=1500, epsilon=-1, objetive=OptimizationObjective.maximize,
                 debug=False, auto_log_each_n_iter=50, experiment_name=None):
        self._opt_factor = objetive.coefficient()
        self.n_iter = n_iter
        self.epsilon = epsilon
        self.fcm = fcm
        self.debug = debug
        self._run_id = uuid4()
        self.auto_log_each_n_iter = auto_log_each_n_iter
        self.experiment_name = experiment_name
        self._results_store = defaultdict(lambda: [])

        self.policies_active = None
        self.ccg = None
        self.graphs_under_policies = []

    def debug_print(self, *args):
        if self.debug:
            print(args)
        else:
            pass

    @abstractmethod
    def _fcm_with_policy(self, mps):
        pass

    def _construct_policies(self):
        self.policies_active = {}
        for idx, (graph, mps) in enumerate(self.graphs_under_policies):
            policy = self._fcm_with_policy(mps)
            fcm_m = PolicyFCM.fcm_with_policy(self.fcm, policy)
            self.policies_active[idx] = (fcm_m, list(policy)[0], mps)

    @abstractmethod
    def step(self):
        trial_id = self._choose_trial()

        fcm_m, policy, mps = self.policies_active[trial_id]
        self.debug_print(f"Policy for {mps.interventional_variables}, {mps.contextual_variables}")

        # logger.debug(f"Trial index {trial_id}")
        self.debug_print(f"Trial index {trial_id}")
        smp = fcm_m.sample(necessary_context=policy.arguments)
        # logger.debug(f"Sample is {smp}")
        self.debug_print(f"Sample is {smp}")

        y = smp[self.ccg.target]
        y = torch.tensor([y])
        return y, policy, smp, mps, trial_id

    def iterate(self, additional_meta_data=None, smoke_test=False):
        stat = datetime.datetime.now()
        for i in tqdm(range(self.n_iter)):
            if i % self.auto_log_each_n_iter == (self.auto_log_each_n_iter-1) and not smoke_test:
                end = datetime.datetime.now()
                self.save_results(stat, end, self.experiment_name, additional_meta_data)
            self.step()
        if not smoke_test:
            end = datetime.datetime.now()
            self.save_results(stat, end, self.experiment_name, additional_meta_data)

    def save_results(self, start, end, prefix="", meta_data: dict = None):

        path = Path(__file__).parent.parent.joinpath("experiment_results")
        if not path.exists():
            path.mkdir()

        path = path.joinpath(prefix)
        if not path.exists():
            path.mkdir()

        path = path.joinpath(f"{prefix}_{self._run_id}.pck")

        with path.open("wb") as fd:
            pickle.dump({'meta': meta_data, "results": dict(self._results_store)}, fd)

    def _choose_trial(self):
        if np.random.uniform(0, 1) < self.epsilon:
            trial_id = np.random.choice(list(self.policies_active.keys()))
            # logger.debug(f"Choosing randomly {trial_id}")
            self.debug_print(f"Choosing randomly {trial_id}")
            return trial_id
        trial_index = None
        try:
            trial_index = [idx for idx, (f, p, _) in (self.policies_active.items()) if p.acq_vals is None][0]
            # logger.info(f"None detected in acquisition function. Choosing {trial_index}")
            self.debug_print(f"None detected in acquisition function. Choosing {trial_index}")
        except IndexError as _:
            fold = np.row_stack([p.acq_vals for f, p, _ in self.policies_active.values()])
            self.debug_print(fold)
            optimal = pareto_optimal(fold)
            self.debug_print(f"Optimal indexes {optimal}")
            trial_index = np.random.choice(optimal)
        return trial_index

    def log_results(self, sample: dict, mps: MixedPolicyScope):
        for k, v in sample.items():
            if type(v) is torch.Tensor:
                v = v.item()
            self._results_store[k].append(v)
        self._results_store['MPS'].append(str(mps))


class POMPSExperiment(Experiment):
    def _fcm_with_policy(self, mps):
        if self.is_single_gp:
            policy = PolicyFCM.mps_to_single_gp(mps, self.factory)
        else:
            policy = PolicyFCM.mps_to_gp_policy(mps, self.factory)
        return policy

    def __init__(self, fcm: FunctionalCausalModel, interventional_variables: tp.Set[str],
                 contextual_variables: tp.Set[str], optimization_domain: tp.List[Domain],
                 target: str = "Y", droppable_scopes: tp.List[MixedPolicyScope] = None, is_single_gp=True,
                 n_iter=1500, epsilon=-1, objetive=OptimizationObjective.maximize, debug=False,
                 auto_log_each_n_iter=50, experiment_name=None):
        super().__init__(fcm, n_iter, epsilon, objetive, debug, auto_log_each_n_iter, experiment_name)
        self._opt_factor = objetive.coefficient()
        self.n_iter = n_iter
        self.epsilon = epsilon
        self.is_single_gp = is_single_gp
        self.fcm = fcm
        self._construct_graphs_under_policy(optimization_domain, interventional_variables, contextual_variables, target)
        self.__drop_undetected(droppable_scopes)
        self._construct_policies()

        self._active_interventional = union([v.interventional_variables for _, _, v in (self.policies_active.values())])
        self._active_context = union([v.contextual_variables for _, _, v in (self.policies_active.values())])
        self.ucb = UCB(len(self.policies_active))

    def _choose_trial(self):
        if np.random.uniform(0, 1) < self.epsilon:
            trial_id = np.random.choice(list(self.policies_active.keys()))
            # logger.debug(f"Choosing randomly {trial_id}")
            self.debug_print(f"Choosing randomly {trial_id}")
            return trial_id
        else:
            return self.ucb.suggest()

    def _construct_graphs_under_policy(self, optimization_domain, interventional_variables,
                                       contextual_variables, target):
        induced = self.fcm.induced_graph()
        self.ccg = ContextualCausalGraph(edges=induced, interventional_variables=interventional_variables,
                                         contextual_variables=contextual_variables, target=target)
        assert {s.name for s in optimization_domain}.issuperset(contextual_variables), \
            "Contextual optimization domain is incomplete"
        assert {s.name for s in optimization_domain}.issuperset(interventional_variables), \
            "Interventional optimization domain is incomplete"
        self.factory = GPFunctorFactory(optimization_domain, acq_function=ReducedMACE)
        simplified = MPSDAGController.simplify(self.ccg)
        self.graphs_under_policies = get_mps_for(simplified)

    def __drop_undetected(self, droppable_scopes):
        droppable_scopes = {} if droppable_scopes is None else droppable_scopes

        scopes = [scope for _, scope in self.graphs_under_policies]
        for droppable in droppable_scopes:
            try:
                idx: int = scopes.index(droppable)
                self.graphs_under_policies[idx] = None
            except ValueError as _:
                continue
        self.graphs_under_policies = list(filter(None, self.graphs_under_policies))

    def step(self):
        y, policy, smp, mps, trial_id = super().step()
        self.ucb.observe(trial_id, -self._opt_factor * y)

        if self.is_single_gp:
            policy.functional.observe(self._opt_factor*y)
        else:
            for p in policy:
                p.functional.observe(self._opt_factor*y)
        self.log_results(smp, mps)

    def save_results(self, start, end, prefix="", meta_data: dict = None):
        md = {"start": start, "end": end, "n_iter": self.n_iter,
              "epsiolon": self.epsilon, "is_single_gp": self.is_single_gp, }
        if meta_data is not None:
            md.update(meta_data)

        super().save_results(start, end, prefix, md)


class CoBOExperiment(POMPSExperiment):

    def _construct_graphs_under_policy(self, optimization_domain, interventional_variables,
                                       contextual_variables, target):
        induced = self.fcm.induced_graph()
        self.ccg = ContextualCausalGraph(edges=induced, interventional_variables=interventional_variables,
                                         contextual_variables=contextual_variables, target=target)
        components = []
        for iv in interventional_variables:
            components.append(PolicyComponent(iv, contextual_variables))
        mps = MixedPolicyScope(set(components))
        assert {s.name for s in optimization_domain}.issuperset(interventional_variables), \
            "Interventional optimization domain is incomplete"
        self.graphs_under_policies = [(MPSDAGController.graph_under_mps(mps, self.ccg), mps) for mps in [mps]]
        self.factory = GPFunctorFactory(optimization_domain, acq_function=ReducedMACE)
