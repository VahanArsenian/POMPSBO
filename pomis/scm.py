import abc
import dataclasses
import enum

import pyro
import typing as tp
from pomis.graphs import NonManGraph, CausalGraph


@dataclasses.dataclass
class Domain:
    name: str

    @abc.abstractmethod
    def get_type(self) -> str:
        pass

    def to_hebo(self) -> tp.Dict:
        return {"type": self.get_type(), "name": self.name}

    @abc.abstractmethod
    def __contains__(self, item):
        pass

    @abc.abstractmethod
    def encode(self, value):
        pass

    @abc.abstractmethod
    def decode(self, value):
        pass

    def __hash__(self):
        return hash(self.name)


@dataclasses.dataclass
class RealDomain(Domain):

    def encode(self, value):
        return value

    def decode(self, value):
        return value

    def __contains__(self, item: float):
        return self.lower_bound <= item <= self.upper_bound

    def get_type(self) -> str:
        return "num"

    def to_hebo(self) -> tp.Dict:
        res = super().to_hebo()
        res.update({"lb": self.lower_bound, "ub": self.upper_bound})
        return res

    lower_bound: float
    upper_bound: float


@dataclasses.dataclass
class BoolDomain(Domain):

    def __contains__(self, item):
        return item in [0, 1]

    def get_type(self) -> str:
        return "bool"

    def to_hebo(self) -> tp.Dict:
        res = super().to_hebo()
        return res

    @classmethod
    def encode(cls, value: bool):
        if value:
            return 1.0
        else:
            return 0.0

    @classmethod
    def decode(cls, value):
        if value == 1:
            return True
        else:
            return False


@dataclasses.dataclass
class IntegralDomain(RealDomain):

    def get_type(self) -> str:
        return "int"

    lower_bound: int
    upper_bound: int


@dataclasses.dataclass
class CategoricalDomain(Domain):

    def __contains__(self, item):
        return item in self.categories

    def get_type(self) -> str:
        return 'cat'

    categories: tp.List


class SCM:

    # TODO: Add graph caching

    def __init__(self, pyro_model: tp.Callable, domains: tp.Union[tp.Iterable[Domain], tp.Sized],
                 non_man=tp.Optional[tp.Set[str]]):
        if non_man is None:
            non_man = set()
        self.non_man = non_man
        assert len(set([dom.name for dom in domains])) == len(domains), "Multiple domains per variable"
        self.domains: tp.Dict[str, Domain] = {dom.name: dom for dom in domains}
        self.sampler = pyro_model
        self.variables = set(self.induced_graph().nodes)

    def induced_graph(self) -> NonManGraph:
        dr = pyro.render_model(self.sampler, model_args=(1,), render_distributions=False)
        start = [i for i, e in enumerate(dr.body) if "}" in e][0] + 1
        edges = ["".join(x.split()).split("->") for x in dr.body[start:]]

        nmg = NonManGraph(self.non_man)
        nmg.add_edges_from(edges)
        return nmg

    def induced_projection(self):
        return CausalGraph(self.induced_graph().projection)

    def domains_for(self, variables: tp.Set[str]):
        return [self.domains[var] for var in variables]

    def hebo_design_space(self, variables: tp.Set[str]):
        domains = self.domains_for(variables)
        return [dom.to_hebo() for dom in domains]

    def do(self, configuration: tp.Dict[str, tp.Any]):
        assert set(configuration.keys()).issubset(self.variables - self.non_man), \
            f"{set(configuration.keys())} is not a valid set of interventional variables"
        assert all([val in self.domains[var] for var, val in configuration.items()]), 'Values outside domain'
        configuration = {var: self.domains[var].encode(val) for var, val in configuration.items()}
        return pyro.poutine.do(self.sampler, data=configuration)
