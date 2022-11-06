from pomis.scm import SCM
from hebo.optimizers.hebo import HEBO
import numpy as np
import pandas as pd
from hebo.design_space.design_space import DesignSpace
import typing as tp


class Objective:

    def __init__(self, scm, maximization=True, number_of_samples=100):
        self.maximize = maximization
        self.scm: SCM = scm
        self.number_of_samples = number_of_samples
        self.objective = self.create_objective()

    @property
    def factor(self):
        return -1 if self.maximize else 1

    def create_objective(self):
        def objective(configuration):
            results = []
            if len(configuration) == 0:
                return (self.factor * self.scm.sampler(self.number_of_samples).mean()).numpy()
            else:
                for config in configuration.to_dict(orient='records'):
                    interventional = self.scm.do(config)
                    results.append(self.factor * interventional(self.number_of_samples).mean())
            return np.array(results).reshape(-1, 1)

        return objective

    def __call__(self, configuration):
        return self.objective(configuration)


class CausalOptimiser:

    @classmethod
    def parse_results(cls, results):
        return pd.DataFrame({"POMIS": results.keys(), "optimas": [v[1] for v in results.values()],
                             "cont_values": [v[0].to_dict(orient='records')[0] for v in results.values()]})

    @classmethod
    def optimise_for(cls, pomis: tp.Set[str], objective: Objective, n_loops, n_suggestions):
        scm: SCM = objective.scm
        if len(pomis) == 0:
            config = pd.DataFrame()
            return pd.DataFrame(data={frozenset(): [dict()]}), objective.factor * objective(config)
        design_space = scm.hebo_design_space(pomis)
        space = DesignSpace().parse(design_space)
        opt = HEBO(space)
        for i in range(n_loops):
            rec = opt.suggest(n_suggestions=n_suggestions)
            opt.observe(rec, objective(rec))
        return opt.best_x, objective.factor * opt.best_y

