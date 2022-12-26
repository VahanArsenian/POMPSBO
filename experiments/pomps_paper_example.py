import logging
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

import argparse

parser = argparse.ArgumentParser(description='Arguments for POMPS example graph')
parser.add_argument('--smoke', action='store_true', help='Used to test the code')
parser.add_argument('--no-smoke', dest='smoke', action='store_false')
parser.set_defaults(smoke=True)
parser.add_argument('--n_iter', type=int, help='Number of iterations to be run', default=1500)
parser.add_argument('--seed', type=int, help='Seed for torch, python, and numpy', default=42)
parser.add_argument('--log_file', type=str, help='Log file path',
                    default="pomps_paper_graph0.log")
parser.add_argument('--experiment_name', type=str, help='Experiment name. Used to define artefact names',
                    default="pomps_paper_graph0")

args = vars(parser.parse_args())
print(args)
smoke_test = args['smoke']
n_iter = args['n_iter']
seed = args['seed']
log_file = args['log_file']
experiment_name = args["experiment_name"]


logger = logging.getLogger('pomps_logger')
logger.setLevel(logging.DEBUG)
# create file handler which logs even debug messages
fh = logging.FileHandler(log_file)
fh.setLevel(logging.DEBUG)
# create console handler with a higher log level
ch = logging.StreamHandler()
ch.setLevel(logging.ERROR)
# create formatter and add it to the handlers
formatter = logging.Formatter('%(process)d-%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
ch.setFormatter(formatter)
fh.setFormatter(formatter)
# add the handlers to logger
logger.addHandler(ch)
logger.addHandler(fh)

import datetime
import pyro
import torch
import pyro.distributions as dist
from pomps.fcm import FunctionalCausalModel, Functor
from experiments.experiment import POMPSExperiment, MixedPolicyScope
from pomis.scm import RealDomain
import numpy as np
import random

np.random.seed(0)
random.seed(0)
torch.manual_seed(0)


def sampler_over():
    u1 = pyro.sample("U1", dist.Uniform(-1, 1))
    u2 = pyro.sample("U2", dist.Uniform(-1, 1))
    return {"U1": u1, "U2": u2}


fcm = FunctionalCausalModel({Functor(lambda U1: pyro.sample("C", dist.Normal(U1, 0.1)), 'C'),
                             Functor(lambda C, U1: pyro.sample("X1", dist.Normal(U1 + C, 0.1)), 'X1'),
                             Functor(
                                 lambda C, X1, U2: pyro.sample("X2", dist.Normal(C + X1 + torch.abs(U2) * 0.3, 0.1)),
                                 'X2'),
                             Functor(lambda U2, X2, C: pyro.sample("Y",
                                                                   dist.Normal(torch.cos(C - X2) + U2 / 100, 0.01)),
                                     "Y")}, sampler_over)

domain = [RealDomain("X1", -2, 2), RealDomain("X2", -2, 2), RealDomain("C", -2, 2)]
start = datetime.datetime.now()

exp = POMPSExperiment(fcm, {"X1", "X2"}, {"C"}, domain, "Y", [MixedPolicyScope(set())],
                      n_iter=16 if smoke_test else n_iter)

exp.iterate()
end = datetime.datetime.now()

if not smoke_test:
    exp.save_results(start, end, "pomps_paper_graph0", {"smoke_test": smoke_test, "seed": seed})

