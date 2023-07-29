from algorithms.causal_bo_experiment import *


class CaBOMABExperiment(CaBOExperiment):

    def __init__(self, fcm: FunctionalCausalModel, non_interventional_variables: tp.Set[str],
                 optimization_domain: tp.List[Domain], target: str = "Y", n_iter=1500, epsilon=-1,
                 objetive=OptimizationObjective.maximize, debug=False, auto_log_each_n_iter=50, experiment_name=None,
                 is_single_gp=True):
        super().__init__(fcm, non_interventional_variables, optimization_domain, target,
                                                n_iter, epsilon, objetive, debug, auto_log_each_n_iter, experiment_name)
        self.ucb = UCB(len(self.policies_active))
        self.is_single_gp = is_single_gp

    def _choose_trial(self):
        return POMPSExperiment._choose_trial(self)

    def step(self):
        y, policy, smp, mps, trial_id = super(CaBOExperiment, self).step()
        self.ucb.observe(trial_id, -self._opt_factor * y)

        policy.functional.observe(self._opt_factor * y)
        self.log_results(smp, mps)
