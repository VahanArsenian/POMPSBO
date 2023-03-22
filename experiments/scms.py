import numpy as np

from pomps.fcm import FunctionalCausalModel
from pomis.scm import Domain, RealDomain
import pyro
import torch
import pyro.distributions as dist
from pomps.fcm import FunctionalCausalModel, Functor
import typing as tp


class SCMOptimizer:

    def __init__(self, scm: FunctionalCausalModel, dm: tp.List[Domain]):
        self.fcm = scm
        self.domain = dm


# ------------------------------------------------------------------------------------------------------------------------------------------------

def latent_sampler_for_aspirin_staitn():
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
                            latent_sampler_for_aspirin_staitn)

domain = [RealDomain("aspirin", 0, 1), RealDomain("statin", 0, 1)]
aspirin_broader_domain = [RealDomain("aspirin", 0, 1), RealDomain("statin", 0, 1),
                          RealDomain('age', 55, 76), RealDomain('bmi', 23, 31)]

aspirin_statin_hom = SCMOptimizer(fcm, domain)
# ------------------------------------------------------------------------------------------------------------------------------------------------
aspirin_statin_hom_broader = SCMOptimizer(fcm, aspirin_broader_domain)

# ------------------------------------------------------------------------------------------------------------------------------------------------
fcm = FunctionalCausalModel({Functor(lambda: pyro.sample("age", dist.Uniform(55, 76)), 'age'),
                             Functor(lambda: pyro.sample("bmi", dist.Normal(27, 0.7)), 'bmi'),
                             Functor(lambda age, bmi: pyro.sample("aspirin",
                                                                  dist.Delta(
                                                                      torch.sigmoid(
                                                                          -8.0 + 0.10 * age + 0.03 * bmi))),
                                     'aspirin'),
                             Functor(lambda age, bmi: pyro.sample("statin", dist.Delta(
                                 torch.sigmoid(-13.0 + 0.10 * age + 0.20 * bmi))),
                                     'statin'),
                             Functor(lambda age, bmi, statin, aspirin: pyro.sample("cancer",
                                                                                   dist.Delta(
                                                                                       statin * statin + torch.square((
                                                                                                                              (
                                                                                                                                      age - 55) / 21) * torch.abs(
                                                                                           (
                                                                                                   bmi - 27) / 4)) + 0.5 * aspirin * aspirin
                                                                                   )),
                                     'cancer'),
                             Functor(lambda age, bmi, statin, aspirin, cancer: pyro.sample("Y", dist.Normal(
                                 cancer + 0.5 * aspirin * aspirin + torch.square(
                                     ((age - 55) / 21) * torch.abs((bmi - 27) / 4)) - 2 * ((age - 55) / 21) * torch.abs(
                                     (bmi - 27) / 4) * (aspirin + statin), 0.01)),
                                     'Y')},
                            latent_sampler_for_aspirin_staitn)

aspirin_statin_het = SCMOptimizer(fcm, domain)
aspirin_statin_het_broader = SCMOptimizer(fcm, aspirin_broader_domain)


# ------------------------------------------------------------------------------------------------------------------------------------------------
def latent_over_pomps_example():
    u1 = pyro.sample("U1", dist.Uniform(-1, 1))
    u2 = pyro.sample("U2", dist.Uniform(-1, 1))
    return {"U1": u1, "U2": u2}


fcm = FunctionalCausalModel({Functor(lambda U1: pyro.sample("C", dist.Normal(U1, 0.1)), 'C'),
                             Functor(lambda U1: pyro.sample("X1", dist.Normal(U1, 0.1)), 'X1'),
                             Functor(
                                     lambda C, X1, U2: pyro.sample("X2", dist.Delta(torch.abs(C - X1) + 0.2*U2)),
                                 'X2'),
                             Functor(lambda U2, X2, C: pyro.sample("Y",
                                                                   dist.Normal(torch.cos(C - X2) + U2 / 10, 0.01)),
                                     "Y")}, latent_over_pomps_example)

domain = [RealDomain("X1", -2, 2), RealDomain("X2", -2, 2), RealDomain("C", -2, 2)]

pomps_example = SCMOptimizer(fcm, domain)
# ------------------------------------------------------------------------------------------------------------------------------------------------
components = [Functor(lambda: pyro.sample("C0", dist.Normal(0, 0.2)), 'C0'),
              Functor(lambda U1, C0: pyro.sample("C1", dist.Normal(C0 - U1, 0.1)), 'C1'),
              Functor(lambda U1: pyro.sample("X1", dist.Normal(U1, 0.1)), 'X1'),

              Functor(lambda C1: pyro.sample("C2", dist.Normal(C1, 0.1)), 'C2'),
              Functor(lambda C2: pyro.sample("C3", dist.Normal(C2, 0.1)), 'C3'),
              Functor(lambda C3: pyro.sample("C4", dist.Normal(C3, 0.1)), 'C4'),
              Functor(lambda C5: pyro.sample("C6", dist.Normal(C5, 0.1)), 'C6'),
              Functor(lambda C1, X1, C6, U2: pyro.sample("X2", dist.Normal(
                  (C1 + C6) / 2 + X1 + torch.abs(U2) * 0.3,
                  0.1)), 'X2'),
              Functor(lambda U2, X2, C1: pyro.sample("Y",
                                                    dist.Normal(torch.cos(C1 - X2) + U2 / 100, 0.01)),
                      "Y")
              ]
n_inter = 27
n_context = 50
for i in range(n_context):
    components.append(Functor(lambda: pyro.sample(f"Ct{i}", dist.Normal(0, 0.2)), f'Ct{i}'))
for i in range(n_inter):
    cc = [c.variable for c in components]
    fn_ = f"""Functor(lambda {", ".join([f"Ct{i}" for i in range(n_context)])}: pyro.sample("Xt{i}", dist.Uniform(-1, 1+0.01*np.mean([{", ".join([f"Ct{i}" for i in range(n_context)])}]))), "Xt{i}")"""
    components.append(eval(fn_))

fn_ = f"""Functor(lambda C4, {", ".join([f"Xt{i}" for i in range(n_inter)])}: pyro.sample("C5", 
dist.Normal(C4+0.001*np.mean([{", ".join([f"Xt{i}" for i in range(n_inter)])}]), 0.1)), 'C5')"""
components.append(eval(fn_))

fcm = FunctionalCausalModel(set(components), latent_over_pomps_example)

domain = [RealDomain("X1", -2, 2), RealDomain("X2", -2, 2)] + [RealDomain(f"Xt{i}", -1, 1) for i in range(n_inter)] +\
         [RealDomain(f'Ct{i}', -1, 1) for i in range(n_context)] + [RealDomain(f'C{i}', -2, 2) for i in range(7)]

pomps_example_hard_for_contextual = SCMOptimizer(fcm, domain)
# ------------------------------------------------------------------------------------------------------------------------------------------------
fcm = FunctionalCausalModel({Functor(lambda U1: pyro.sample("C", dist.Normal(U1, 0.1)), 'C'),
                             Functor(lambda U1: pyro.sample("X1", dist.Normal(U1, 0.1)), 'X1'),
                             Functor(
                                 lambda C, X1, U2: pyro.sample("X2", dist.Normal(C + X1, 0.01)),
                                 'X2'),
                             Functor(lambda U2, X2, C: pyro.sample("Y",
                                                                   dist.Normal(torch.cos(C - X2) + U2 / 100, 0.001)),
                                     "Y")}, latent_over_pomps_example)

pomps_both_optimal_example = SCMOptimizer(fcm, domain)
# ------------------------------------------------------------------------------------------------------------------------------------------------
fcm = FunctionalCausalModel({Functor(lambda U1: pyro.sample("C", dist.Delta(U1)), 'C'),
                             Functor(lambda U1: pyro.sample("X1", dist.Delta(U1)), 'X1'),
                             Functor(
                                 lambda C, X1, U2: pyro.sample("X2", dist.Delta(U2 * torch.exp(-(X1 + C) * (X1 + C)))),
                                 'X2'),
                             Functor(lambda U2, X2, C: pyro.sample("Y",
                                                                   dist.Delta(U2 * X2 + 0.01 * C)),
                                     "Y")}, latent_over_pomps_example)

domain = [RealDomain("X1", -1, 1), RealDomain("X2", -2, 2), RealDomain("C", -1, 1)]

impossible_for_contextual_bo = SCMOptimizer(fcm, domain)
# ------------------------------------------------------------------------------------------------------------------------------------------------
