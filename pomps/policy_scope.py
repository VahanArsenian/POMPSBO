import typing as tp
from pomps.utils import union


class PolicyComponent:

    def __init__(self, target: str, contexts: tp.Set[str]):
        self.target = target
        self.context = contexts

    def to_pairs(self) -> tp.Set[tp.Tuple[str, str]]:
        return set([(context, self.target) for context in self.context])

    def __hash__(self):
        return self.target

    def __eq__(self, other: "PolicyComponent"):
        if self.target == other.target and self.context == other.context:
            return True
        else:
            return False

    def __ge__(self, other: "PolicyComponent"):
        if self.target == other.target and other.context.issubset(self.context):
            return True
        else:
            return False


class MixedPolicyScope:

    def __init__(self, components: tp.Set[PolicyComponent]):
        self.components: tp.Dict[str, PolicyComponent] = {component.target: component for component in components}

    @property
    def interventional_variables(self):
        return set(self.components.keys())

    @property
    def contextual_variables(self):
        return union([c.context for c in self.components.values()])

    @property
    def pairs(self):
        return union([component.to_pairs() for component in self.components.values()])
