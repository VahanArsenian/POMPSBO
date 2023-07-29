import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from experiments.scms import aspirin_statin_hom as environ
import datetime
import torch
from algorithms.causal_bo_mab_experiment import CaBOMABExperiment, OptimizationObjective
import numpy as np
import random


if __name__ == "__main__":
    start = datetime.datetime.now()

    from experiments.utils import *

    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    experiment_name = "cbo_mab_aspirin_statin_hom"

    exp = CaBOMABExperiment(environ.fcm, {"age", "bmi", "cancer"},
                         environ.domain, "Y", n_iter=16 if smoke_test else n_iter,
                         experiment_name=experiment_name, objetive=OptimizationObjective.minimize)

    exp.iterate({"smoke_test": smoke_test, "seed": seed}, smoke_test=smoke_test)
    end = datetime.datetime.now()

