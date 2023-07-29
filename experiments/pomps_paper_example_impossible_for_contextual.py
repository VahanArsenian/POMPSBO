import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from experiments.scms import impossible_for_contextual_bo as environ
import datetime
import torch
from algorithms.pomps_experiment import POMPSExperiment, MixedPolicyScope
import numpy as np
import random


if __name__ == "__main__":
    start = datetime.datetime.now()

    from experiments.utils import *

    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    experiment_name = "pomps_paper_graph0_impossible_for_contextual"

    exp = POMPSExperiment(environ.fcm, {"X1", "X2"}, {"C"}, environ.domain,
                          "Y", [MixedPolicyScope(set())], debug=smoke_test,
                          n_iter=16 if smoke_test else n_iter, experiment_name=experiment_name)

    exp.iterate({"smoke_test": smoke_test, "seed": seed}, smoke_test=smoke_test)
    end = datetime.datetime.now()



