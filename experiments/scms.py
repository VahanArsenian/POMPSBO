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
                                                                                   dist.Delta(torch.sigmoid(
                                                                                       (
                                                                                        bmi * aspirin + age * statin) /
                                                                                       (aspirin + statin + 1e-8)
                                                                                   ))),
                                     'cancer'),
                             Functor(lambda age, bmi, statin, aspirin, cancer: pyro.sample("Y", dist.Normal(
                                  cancer/torch.sigmoid(bmi)+torch.square(aspirin-(age-55)/76)+statin, 0.01)),
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
                             Functor(lambda C, U1: pyro.sample("X1", dist.Normal(U1 + C, 0.1)), 'X1'),
                             Functor(
                                 lambda C, X1, U2: pyro.sample("X2", dist.Normal(torch.abs(C - X1) + 0.2, 0.1)),
                                 'X2'),
                             Functor(lambda U2, X2, C: pyro.sample("Y",
                                                                   dist.Normal(torch.cos(C - X2) + U2 / 100, 0.001)),
                                     "Y")}, latent_over_pomps_example)

domain = [RealDomain("X1", -2, 2), RealDomain("X2", -2, 2), RealDomain("C", -2, 2)]

pomps_example = SCMOptimizer(fcm, domain)
# ------------------------------------------------------------------------------------------------------------------------------------------------
fcm = FunctionalCausalModel({Functor(lambda U1: pyro.sample("C", dist.Normal(U1, 0.1)), 'C'),
                             Functor(lambda C, U1: pyro.sample("X1", dist.Normal(U1 + C, 0.1)), 'X1'),
                             Functor(
                                 lambda C, X1, U2: pyro.sample("X2", dist.Normal(C + X1, 0.01)),
                                 'X2'),
                             Functor(lambda U2, X2, C: pyro.sample("Y",
                                                                   dist.Normal(torch.cos(C - X2) + U2 / 100, 0.001)),
                                     "Y")}, latent_over_pomps_example)

pomps_both_optimal_example = SCMOptimizer(fcm, domain)
# ------------------------------------------------------------------------------------------------------------------------------------------------
