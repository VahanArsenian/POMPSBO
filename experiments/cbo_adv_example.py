import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from experiments.scms import adv_example as environ
import datetime
import torch
from algorithms.causal_bo_experiment import CaBOExperiment, OptimizationObjective
import numpy as np
import random


if __name__ == "__main__":
    start = datetime.datetime.now()

    from experiments.utils import *

    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    experiment_name = "cbo_adv_example"

    exp = CaBOExperiment(environ.fcm, {'clicks', 'device', 'country', 'RT', 'UA', 'probability', 'day_part',
                                       'connection_type', 'SK', 'installs', 'category', 'impressions', 'bid_price',
                                       'purchases', 'user_profile', 'exchange'},
                         environ.domain, "revenue", n_iter=16 if smoke_test else n_iter,
                         experiment_name=experiment_name, objetive=OptimizationObjective.maximize)

    exp.iterate({"smoke_test": smoke_test, "seed": seed}, smoke_test=smoke_test)
    end = datetime.datetime.now()

