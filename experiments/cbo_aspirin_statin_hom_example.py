import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
import datetime
import pyro
import torch
import pyro.distributions as dist
from pomps.fcm import FunctionalCausalModel, Functor
from experiments.causal_bo_experiment import CaBOExperiment, OptimizationObjective
from pomis.scm import BoolDomain, RealDomain
import numpy as np
import random


def sampler_over():
    return dict()


fcm = FunctionalCausalModel({Functor(lambda: pyro.sample("age", dist.Uniform(55, 76)), 'age'),
                             Functor(lambda: pyro.sample("bmi", dist.Normal(27, 0.7)), 'bmi'),
                             Functor(lambda age, bmi: pyro.sample("aspirin",
                                                                  dist.Delta(
                                                                      torch.sigmoid(-8.0 + 0.10 * age + 0.03 * bmi))),
                                     'aspirin'),
                             Functor(lambda age, bmi: pyro.sample("statin", dist.Delta(
                                 torch.sigmoid(-13.0 + 0.10 * age + 0.20 * bmi))),
                                     'statin'),
                             Functor(lambda age, bmi, statin, aspirin: pyro.sample("cancer",
                                                                                   dist.Delta(torch.sigmoid(
                                                                                       2.2 - 0.05 * age
                                                                                       + 0.01 * bmi - 0.04 * statin
                                                                                       + 0.02 * aspirin))),
                                     'cancer'),
                             Functor(lambda age, bmi, statin, aspirin, cancer: pyro.sample("Y", dist.Normal(
                                 6.8 + 0.04 * age - 0.15 * bmi - 0.60 * statin + 0.55 * aspirin + 1.00 * cancer, 0.4)),
                                     'Y')},
                            sampler_over)

domain = [RealDomain("aspirin", 0, 1), RealDomain("statin", 0, 1)]
# domain = [BoolDomain('aspirin'), BoolDomain('statin'), RealDomain('age', 55, 76), RealDomain('bmi', 23, 31)]


if __name__ == "__main__":
    start = datetime.datetime.now()

    from experiments.utils import *

    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    experiment_name = "cbo_"

    exp = CaBOExperiment(fcm, {"age", "bmi", "cancer"}, domain, "Y", n_iter=16 if smoke_test else n_iter,
                         experiment_name=experiment_name, objetive=OptimizationObjective.minimize)

    exp.iterate({"smoke_test": smoke_test, "seed": seed}, smoke_test=smoke_test)
    end = datetime.datetime.now()

