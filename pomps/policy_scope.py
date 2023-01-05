import copy
import typing as tp
from pomps.utils import union


class PolicyComponent:

    def __init__(self, target: str, contexts: tp.Set[str]):
        self.target = target
        self.context = contexts

    def to_pairs(self) -> tp.Set[tp.Tuple[str, str]]:
        return set([(context, self.target) for context in self.context])

    def __hash__(self):
        return hash(self.target)

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

    def __repr__(self):
        return f"<{self.target}, {sorted(self.context)}>"


class MixedPolicyScope:

    def __init__(self, components: tp.Set[PolicyComponent]):
        self.components: tp.Dict[str, PolicyComponent] = {component.target: component for component in components}

    @property
    def interventional_variables(self):
        return set(self.components.keys())

    @property
    def contextual_variables(self):
        return union([c.context for c in self.components.values()])

    def __eq__(self, other: "MixedPolicyScope"):
        if other.interventional_variables != self.interventional_variables:
            return False
        for component in self.components:
            self_component = self.components[component]
            other_component = other.components[component]
            if self_component.context != other_component.context:
                return False
        return True

    @property
    def pairs(self):
        if len(self.components) == 0:
            return set()
        return union([component.to_pairs() for component in self.components.values()])

    def __repr__(self):
        return "\t".join([v.__repr__() for v in sorted(list(self.components.values()), key=lambda x: x.target)])

    def implied(self, variables: tp.Set[str]):
        # print(variables)
        result = copy.copy(variables)
        future_result = copy.copy(variables)
        should_repeat = True
        # print('call')
        while should_repeat:
            for target, component in self.components.items():
                if component.context.issubset(future_result):
                    future_result = future_result | {target}
                    # print(future_result, result)
            should_repeat = future_result != result
            result = future_result
        return result
