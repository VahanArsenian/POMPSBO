import numpy as np

from pomps.hebo_adapted import AdHEBO, HEBO, ReducedMACE
import pandas as pd
import typing as tp
from pomps.fcm import Functor
from pomis.scm import Domain
import torch
from hebo.design_space.design_space import DesignSpace
from pomps.utils import union


class GPFunctional:

    def __init__(self, optimizer: HEBO):
        self.optimizer = optimizer
        self.__x: pd.DataFrame = None

    def __call__(self, **kwargs):
        suggested, acq_vals = self.optimizer.suggest(1, fix_input=kwargs)
        self.__x = suggested
        self.acq_vals = acq_vals
        result: tp.Dict = suggested.to_dict(orient="records")[0]
        for k in kwargs:
            del result[k]
        return result, acq_vals

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
        self.acq_vals = None

    def __call__(self, payload: tp.Dict[str, tp.Any]):
        result, self.acq_vals = super(GPFunctor, self).__call__(payload)
        return torch.tensor(list(result.values())[0])


class SFBuffer:

    def __init__(self):
        self.buffer: dict = {}

    def register(self, sfu: 'SharedFunctor'):
        sfu.sf = self


class SharedFunctor(GPFunctor):
    def __init__(self, functional: GPFunctional, variable, arguments: tp.Set[str], sfb: SFBuffer = None):
        super().__init__(functional, variable)
        self.hyper_space = self.arguments
        self.arguments = arguments
        self.sf = sfb

    def __call__(self, payload: tp.Dict[str, tp.Any]):
        # print("buffer", self.sf.buffer)
        if len(self.sf.buffer) == 0:
            assert set(payload.keys()).issuperset(self.arguments), "Signature mismatch"
            active_payload = {k: v for k, v in payload.items() if k in self.arguments}
            # print(active_payload, payload, self.hyper_space, self.arguments, self.variable)
            self.sf.buffer, _ = self.functional(**active_payload)
        self.acq_vals = self.functional.acq_vals
        return self.sf.buffer.pop(self.variable)


class GPFunctorFactory:

    def __init__(self, domains: tp.List[Domain], acq_function=ReducedMACE):
        self.domains: tp.Dict[str, Domain] = {dom.name: dom for dom in domains}
        self.acq_func = acq_function

    def get_domain_for(self, variables: tp.Set[str]) -> tp.List[Domain]:
        return [v for k, v in self.domains.items() if k in variables]

    def construct(self, variable, arguments: tp.Set) -> GPFunctor:
        domain = self.get_domain_for({variable} | arguments)
        hebo_space = [d.to_hebo() for d in domain]
        ds = DesignSpace().parse(hebo_space)

        functional = GPFunctional(AdHEBO(ds))
        return GPFunctor(functional, variable)

    def construct_shared(self, variables: tp.List[str], arguments: tp.List[tp.Set[str]]) -> tp.List[GPFunctor]:
        buffer = SFBuffer()
        # print(arguments)
        domain = self.get_domain_for(set(variables) | union(arguments))
        hebo_space = [d.to_hebo() for d in domain]
        ds = DesignSpace().parse(hebo_space)
        hebo = GPFunctional(AdHEBO(ds, acq_cls=self.acq_func))
        functional = [SharedFunctor(hebo, v, c) for v, c in zip(variables, arguments)]
        [buffer.register(f) for f in functional]
        return functional
