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
                                 lambda C, X1, U2: pyro.sample("X2", dist.Normal(torch.abs(C - X1) + 0.2, 0.1)),
                                 'X2'),
                             Functor(lambda U2, X2, C: pyro.sample("Y",
                                                                   dist.Normal(torch.cos(C - X2) + U2 / 100, 0.001)),
                                     "Y")}, latent_over_pomps_example)

domain = [RealDomain("X1", -2, 2), RealDomain("X2", -2, 2), RealDomain("C", -2, 2)]

pomps_example = SCMOptimizer(fcm, domain)
# ------------------------------------------------------------------------------------------------------------------------------------------------
fcm = FunctionalCausalModel({Functor(lambda: pyro.sample("C0", dist.Normal(0, 0.2)), 'C0'),
                             Functor(lambda U1, C0: pyro.sample("C", dist.Normal(C0 - U1, 0.1)), 'C'),
                             Functor(lambda U1: pyro.sample("X1", dist.Normal(U1, 0.1)), 'X1'),
                             Functor(lambda: pyro.sample("X3", dist.Uniform(-1, 1)), 'X3'),
                             Functor(lambda C, X3: pyro.sample("C2", dist.Normal(C+0.001*X3, 0.1)), 'C2'),
                             Functor(lambda C2: pyro.sample("C3", dist.Normal(C2, 0.1)), 'C3'),
                             Functor(lambda: pyro.sample("X4", dist.Uniform(-1, 1)), 'X4'),
                             Functor(lambda C3, X4: pyro.sample("C4", dist.Normal(C3+0.001*X4, 0.1)), 'C4'),
                             Functor(lambda: pyro.sample("X5", dist.Uniform(-1, 1)), 'X5'),
                             Functor(lambda C4, X5: pyro.sample("C5", dist.Normal(C4+0.001*X5, 0.1)), 'C5'),
                             Functor(lambda C5: pyro.sample("C6", dist.Normal(C5, 0.1)), 'C6'),
                             Functor(lambda C, X1, C6, U2: pyro.sample("X2", dist.Normal(
                                 (C + C6) / 2 + X1 + torch.abs(U2) * 0.3,
                                 0.1)), 'X2'),
                             Functor(lambda U2, X2, C: pyro.sample("Y",
                                                                   dist.Normal(torch.cos(C - X2) + U2 / 100, 0.01)),
                                     "Y")}, latent_over_pomps_example)

domain = [RealDomain("X1", -2, 2), RealDomain("X2", -2, 2), RealDomain("X3", -1, 1), RealDomain("X4", -1, 1),
          RealDomain("X5", -1, 1), RealDomain("C", -2, 2), RealDomain("C0", -2.2, 2.2),
          RealDomain("C2", -2.4, 2.4), RealDomain("C3", -2.6, 2.6), RealDomain("C4", -2.8, 2.8),
          RealDomain("C5", -3, 3),
          RealDomain("C6", -3.2, 3.2)]

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
                                 lambda C, X1, U2: pyro.sample("X2", dist.Delta(U2*torch.exp(-(X1+C)*(X1+C)))),
                                 'X2'),
                             Functor(lambda U2, X2, C: pyro.sample("Y",
                                                                   dist.Delta(U2*X2+0.01*C)),
                                     "Y")}, latent_over_pomps_example)

domain = [RealDomain("X1", -1, 1), RealDomain("X2", -2, 2), RealDomain("C", -1, 1)]

impossible_for_contextual_bo = SCMOptimizer(fcm,  domain)
# ------------------------------------------------------------------------------------------------------------------------------------------------
