from hebo.optimizers.hebo import HEBO
import pandas as pd
import typing as tp
from pomps.fcm import Functor
from pomis.scm import Domain
import torch
from hebo.design_space.design_space import DesignSpace


class GPFunctional:

    def __init__(self, optimizer: HEBO):
        self.optimizer = optimizer
        self.__x: pd.DataFrame = None

    def __call__(self, **kwargs):
        suggested: pd.DataFrame = self.optimizer.suggest(1, fix_input=kwargs)
        self.__x = suggested
        result: tp.Dict = suggested.to_dict(orient="records")[0]
        for k in kwargs:
            del result[k]
        return torch.tensor(list(result.values())[0])

    def observe(self, target):
        self.optimizer.observe(self.__x, target)
        self.__x = None

    def suggest(self):
        pass


class GPFunctor(Functor):

    def __init__(self, functional: GPFunctional, variable):
        self.variable = variable
        self.functional = functional
        self.arguments = set(functional.optimizer.space.para_names) - {self.variable}


class GPFunctorFactory:

    def __init__(self, domains: tp.List[Domain]):
        self.domains: tp.Dict[str, Domain] = {dom.name: dom for dom in domains}

    def get_domain_for(self, variables: tp.Set[str]) -> tp.List[Domain]:
        return [v for k, v in self.domains.items() if k in variables]

    def construct(self, variable, arguments: tp.Set) -> GPFunctor:
        domain = self.get_domain_for({variable} | arguments)
        hebo_space = [d.to_hebo() for d in domain]
        ds = DesignSpace().parse(hebo_space)

        functional = GPFunctional(HEBO(ds))
        return GPFunctor(functional, variable)