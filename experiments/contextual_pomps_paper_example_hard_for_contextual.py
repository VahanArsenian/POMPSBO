import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from experiments.scms import pomps_example_hard_for_contextual as environ
import datetime
import torch
from experiments.pomps_experiment import CoBOExperiment, MixedPolicyScope
import numpy as np
import random


if __name__ == "__main__":
    start = datetime.datetime.now()

    from experiments.utils import *

    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    experiment_name = "contextual_pomps_paper_graph0_hard_for_contextual"

    exp = CoBOExperiment(environ.fcm, {"X1", "X2", "X3", "X4", "X5"}, {"C", "C2", 'C3', 'C0', "C4", 'C5', 'C6'},
                         environ.domain, "Y", [MixedPolicyScope(set())], debug=smoke_test,
                         n_iter=16 if smoke_test else n_iter, experiment_name=experiment_name)

    exp.iterate({"smoke_test": smoke_test, "seed": seed}, smoke_test=smoke_test)
    end = datetime.datetime.now()



