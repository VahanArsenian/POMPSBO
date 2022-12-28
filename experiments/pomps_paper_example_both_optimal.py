import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from experiments.utils import *
import datetime
import pyro
import torch
import pyro.distributions as dist
from pomps.fcm import FunctionalCausalModel, Functor
from experiments.experiment import POMPSExperiment, MixedPolicyScope
from pomis.scm import RealDomain
import numpy as np
import random

np.random.seed(seed)
random.seed(seed)
torch.manual_seed(seed)


def sampler_over():
    u1 = pyro.sample("U1", dist.Uniform(-1, 1))
    u2 = pyro.sample("U2", dist.Uniform(-1, 1))
    return {"U1": u1, "U2": u2}


fcm = FunctionalCausalModel({Functor(lambda U1: pyro.sample("C", dist.Normal(U1, 0.1)), 'C'),
                             Functor(lambda C, U1: pyro.sample("X1", dist.Normal(U1 + C, 0.1)), 'X1'),
                             Functor(
                                 lambda C, X1, U2: pyro.sample("X2", dist.Normal(C+X1, 0.01)),
                                 'X2'),
                             Functor(lambda U2, X2, C: pyro.sample("Y",
                                                                   dist.Normal(torch.cos(C - X2) + U2 / 100, 0.001)),
                                     "Y")}, sampler_over)

domain = [RealDomain("X1", -2, 2), RealDomain("X2", -2, 2), RealDomain("C", -2, 2)]
start = datetime.datetime.now()

exp = POMPSExperiment(fcm, {"X1", "X2"}, {"C"}, domain, "Y", [MixedPolicyScope(set())],
                      n_iter=16 if smoke_test else n_iter)

exp.iterate()
end = datetime.datetime.now()

if not smoke_test:
    exp.save_results(start, end, "pomps_paper_graph0", {"smoke_test": smoke_test, "seed": seed})

