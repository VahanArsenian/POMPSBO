import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from experiments.scms import aspirin_statin_het_broader as environ
import datetime
import torch
from algorithms.pomps_experiment import POMPSExperiment, OptimizationObjective
from pomps.policy_scope import PolicyComponent, MixedPolicyScope
import numpy as np
import random


if __name__ == "__main__":
    start = datetime.datetime.now()

    from experiments.utils import *

    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    experiment_name = "pomps_aspirin_statin_het"

    exp = POMPSExperiment(environ.fcm, {"aspirin", 'statin'}, {'age', "bmi"},
                          environ.domain, n_iter=16 if smoke_test else n_iter,
                          objetive=OptimizationObjective.minimize,
                          droppable_scopes=[MixedPolicyScope(set()),
                                            MixedPolicyScope({PolicyComponent('aspirin', {'age', 'bmi'})}),
                                            MixedPolicyScope({PolicyComponent('statin', {'age', 'bmi'})})],
                          experiment_name=experiment_name)

    exp.iterate({"smoke_test": smoke_test, "seed": seed}, smoke_test=smoke_test)
    end = datetime.datetime.now()

