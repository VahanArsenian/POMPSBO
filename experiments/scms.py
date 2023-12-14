import numpy as np

from pomps.fcm import FunctionalCausalModel
from pomis.scm import Domain, RealDomain, BoolDomain
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

def latents():
    return {}
    u1 = pyro.sample("error5", dist.Uniform(-1, 1))
    u2 = pyro.sample("error2", dist.Uniform(-1, 1))
    u3 = pyro.sample("app_views_count", dist.Poisson(100))
    return {"error5": u1, 'error2': u2, 'app_views_count': u3}


std = 0.42


def country_to_con_t_logits(country):
    return -0.9362607 + 0.4905761 * country


def country_to_day_p_logits(country):
    return 0.1089 + 0.9141 * country


def to_device(country, connection_type):
    return -0.5022 - 1.5922 * country - 0.1550 * connection_type


def to_exchange(device, day_part, country):
    return -34045.091 * device - 2973.584 * day_part + 54712.746 * country - 34045.091 * (
                device ** 2) + 365.688 * device * day_part \
           - 53906.752 * device * country + 570.279 * (day_part ** 2) + 1667.335 * day_part * country + 54712.746 * (
                       country ** 2)


def to_budget(country, exchange):
    return 397.04 * country + 0.00375 * exchange + 397.042 * (country ** 2) + 0.00043 * country * exchange - 5.9e-09 * (
                exchange ** 2)


def to_ad_type(category, connection_type, exchange):
    return 0.8619 + 2.2977 * exchange - 0.3002 * connection_type + 2.8386 * category


def to_creative(category, device, ad_type):
    return -0.00196 * category - 8980.50169 * device + 53.20869 * ad_type - 0.0 * (
                category ** 2) + 0.42262 * category * device \
           + 0.19588 * category * ad_type - 8980.50169 * (device ** 2) + 12626.37185 * device * ad_type + 53.20869 * (
                       ad_type ** 2) \
           + 8.191084166028143e-09


def to_user_profile(RT, SK, UA, ad_type, creative, country, exchange):
    return -6.216e-07 + 9.83 * RT - 6.6 * SK - 3.23 * UA - 1.32 * ad_type - 0.0006 * creative - 0.86 * country - 6.2778e-06 * exchange + 9.8 * (
                RT ** 2) - 3e-10 * RT * SK - 1e-10 * RT * UA - 9.23 * RT * ad_type - 0.0016 * RT * creative + 11.55 * RT * country \
           - 4.9848e-06 * RT * exchange - 6.6 * (
                       SK ** 2) + 3.3 * SK * ad_type + 0.00059 * SK * creative - 3.7 * SK * country + 1.4834e-06 * SK * exchange - 3.23 * (
                       UA ** 2) + 4.6 * UA * ad_type + 0.0005 * UA * creative - 8.7 * UA * country - 2.7679e-06 * UA * exchange \
           - 1.32 * (
                       ad_type ** 2) - 3.35407e-05 * ad_type * creative - 5.7 * ad_type * country + 5.4636e-06 * ad_type * exchange - 1e-10 * (
                       creative ** 2) + 7.60864e-05 * creative * country + 1e-10 * creative * exchange - 0.86 * (
                       country ** 2) \
           + 2.13998e-05 * country * exchange - 0.0 * (exchange ** 2) + 2.020344363166225e-11


def to_goal(exchange, country):
    return -7.27945e-05 * exchange + 35.4976 * country + 1e-10 * (
                exchange ** 2) - 9.24829e-05 * exchange * country + 35.5 * (country ** 2)


def to_probability(category, ad_type, user_profile, day_part, country, connection_type, exchange):
    return 3.6e-09 - 3.4e-09 * category - 0.0017 * ad_type - 4.2267e-06 * user_profile - 8.96551e-05 * day_part - 0.00076 * country - 3.22984e-05 * connection_type + 1e-10 * exchange + 0.0 * (
                category ** 2) + 2.6e-09 * category * ad_type \
           + 0.0 * category * user_profile + 0.0 * category * day_part + 1e-10 * category * country + 1e-10 * category * connection_type - 0.0 * category * exchange - 0.00173 * (
                       ad_type ** 2) + 3.1211e-06 * ad_type * user_profile + 4.27544e-05 * ad_type * day_part \
           + 0.00163 * ad_type * country + 0.00035 * ad_type * connection_type - 7e-10 * ad_type * exchange + 1.7e-09 * (
                       user_profile ** 2) + 2.27e-08 * user_profile * day_part + 1.9398e-06 * user_profile * country - 2.991e-07 * user_profile * connection_type \
           - 0.0 * user_profile * exchange + 5.0122e-06 * (day_part ** 2) \
           + 3.3536e-06 * day_part * country + 1.3187e-05 * day_part * connection_type + 0.0 * day_part * exchange - 0.00076 * (
                       country ** 2) - 7.76586e-05 * country * connection_type - 4e-10 * country * exchange - 9.68951e-05 * (
                       connection_type ** 2) \
           - 1e-10 * connection_type * exchange + 0.0 * (exchange ** 2)


def to_bid_price(goal, budget, ad_type, day_part, exchange, country, probability):
    return -0.004 - 0.037 * goal - 0.003 * budget - 9.882 * ad_type - 0.593 * day_part - 0.0 * exchange - 11.343 * country - 770.665 * probability + 0.0 * (
                goal ** 2) - 0.0 * goal * budget - 0.007 * goal * ad_type + 0.001 * goal * day_part - 0.0 * goal * exchange + 0.051 * goal * country \
           + 2.318 * goal * probability + 0.0 * (
                       budget ** 2) - 0.001 * budget * ad_type + 0.0 * budget * day_part + 0.0 * budget * exchange + 0.001 * budget * country + 0.512 * budget * probability - 9.863 * (
                       ad_type ** 2) + 0.5 * ad_type * day_part + 0.0 * ad_type * exchange \
           + 10.304 * ad_type * country + 553.419 * ad_type * probability - 0.128 * (
                       day_part ** 2) + 0.0 * day_part * exchange + 0.696 * day_part * country - 20.468 * day_part * probability - 0.0 * (
                       exchange ** 2) + 0.0 * exchange * country + 0.001 * exchange * probability \
           - 11.342 * (country ** 2) + 826.853 * country * probability + 491.348 * (
                       probability ** 2) - 2.049786790812473e-11


def to_impressions(bid_price, day_part, user_profile, ad_type, country, creative, category, exchange):     return (
            -5.5663 + 6.9396 * bid_price - 0.0954 * day_part + 6.7361 * user_profile + 1.2766 * ad_type - 0.0780 * country + 0.0173 * creative + 0.8825 * category - 0.9427 * exchange)


def to_click(exchange, creative, ad_type, country, user_profile, probability, impressions, ):
    return (
                -1.9162808 + 0.4774666 * exchange - 1.6800447 * creative - 5.0000760 * ad_type - 0.3616471 * country - 4.5300518 * user_profile + 11.2189913 * probability + 4.6414798 * impressions)


def to_installs(category, creative, ad_type, country, user_profile, probability, impressions, clicks, exchange):
    return (
                -6.3968 - 2.0413 * category - 0.0523 * impressions - 3.6981 * user_profile + 11.1342 * probability + 14.8660 * clicks - 3.8073 * creative - 1.1441 * country - 0.7202 * exchange)


def to_purchases(ad_type, installs, country, clicks, category, device, user_profile, creative, impressions, probability,
                 exchange):
    return (
                -13.1807 - 0.8743 * ad_type + 2.9148 * installs - 0.2362 * country + 0.0125 * clicks - 0.4308 * category + 1.1486 * device - 0.0320 * user_profile - 0.2640 * creative - 0.0001 * impressions + 0.0067 * probability - 0.2261 * exchange)


def to_revenue(budget, purchases, bid_price, country, exchange, category):
    return 3.6e-09 - 3.4e-09 * budget - 0.0017334935 * purchases - 4.2267e-06 * bid_price - 8.96551e-05 * country - 0.0007567844 * exchange - 3.22984e-05 * category + 1e-10 * (
                budget ** 2) + 0.0 * budget * purchases + 2.6e-09 * budget * bid_price \
           + 0.0 * budget * country + 0.0 * budget * exchange + 1e-10 * budget * category + 1e-10 * (
                       purchases ** 2) - 0.0 * purchases * bid_price - 0.0017 * purchases * country + 3.1211e-06 * purchases * exchange + 4.27544e-05 * purchases * category \
           + 0.0016 * (
                       bid_price ** 2) + 0.00035 * bid_price * country - 7e-10 * bid_price * exchange + 1.7e-09 * bid_price * category + 2.27e-08 * (
                       country ** 2) + 1.9398e-06 * country * exchange - 2.991e-07 * country * category - 0.0 * (
                       exchange ** 2) \
           + 5.0122e-06 * exchange * category + 3.3536e-06 * (category ** 2)


adv_scm = FunctionalCausalModel({
    Functor(lambda: pyro.sample("country", dist.Categorical(probs=torch.tensor([0.5,0.5]))), 'country'),
    Functor(lambda: pyro.sample("category", dist.Categorical(probs=torch.tensor([0.5,0.5]))), 'category'),
    Functor(lambda: pyro.sample("RT", dist.Bernoulli(probs=torch.tensor(0.5))), 'RT'),
    Functor(lambda: pyro.sample("SK", dist.Bernoulli(probs=torch.tensor(0.5))), 'SK'),
    Functor(lambda: pyro.sample("UA", dist.Bernoulli(probs=torch.tensor(0.5))), 'UA'),
    Functor(lambda country: pyro.sample("connection_type", dist.Bernoulli(logits=(country_to_con_t_logits(country)))), 'connection_type'),
    Functor(lambda country: pyro.sample("day_part", dist.Bernoulli(logits=(country_to_day_p_logits(country)))), 'day_part'),
    Functor(lambda country, connection_type: pyro.sample("device", dist.Bernoulli(logits=(to_device(country, connection_type)))), 'device'),
    Functor(lambda device, day_part, country: pyro.sample("exchange", dist.Normal(to_exchange(device, day_part, country), std)), 'exchange'),
    Functor(lambda country, exchange: pyro.sample("budget", dist.Normal(to_budget(country, exchange), std)), 'budget'),
    Functor(lambda category, connection_type, exchange: pyro.sample("ad_type", dist.Bernoulli(logits=to_ad_type(category, connection_type, exchange))), 'ad_type'),
    Functor(lambda category, device, ad_type: pyro.sample("creative", dist.Normal(to_creative(category, device, ad_type), std)), 'creative'),
    Functor(lambda RT, SK, UA, ad_type, creative, country, exchange: pyro.sample("user_profile", dist.Normal(to_user_profile(RT, SK, UA, ad_type, creative, country, exchange), std)), 'user_profile'),
    Functor(lambda exchange, country: pyro.sample("goal", dist.Normal(to_goal(exchange, country), std)), 'goal'),
    Functor(lambda category, ad_type, user_profile, day_part, country, connection_type, exchange: pyro.sample("probability", dist.Normal(to_probability(category, ad_type, user_profile, day_part, country, connection_type, exchange), std)), 'probability'),
    Functor(lambda goal, budget, ad_type, day_part, exchange, country, probability: pyro.sample("bid_price", dist.Normal(to_bid_price(goal, budget, ad_type, day_part, exchange, country, probability), std)), 'bid_price'),
    Functor(lambda bid_price, day_part, user_profile, ad_type, country, creative, category, exchange: pyro.sample("impressions", dist.Bernoulli(logits=to_impressions(bid_price, day_part, user_profile, ad_type, country, creative, category, exchange))), 'impressions'),
    Functor(lambda exchange, creative, ad_type, country, user_profile, probability, impressions: pyro.sample("clicks", dist.Bernoulli(logits=to_click(exchange, creative, ad_type, country, user_profile, probability, impressions))), 'clicks'),
    Functor(lambda category, creative, ad_type, country, user_profile, probability, impressions, clicks, exchange: pyro.sample("installs", dist.Bernoulli(logits=to_installs(category, creative, ad_type, country, user_profile, probability, impressions, clicks, exchange))), 'installs'),
    Functor(lambda category, creative, ad_type, country, user_profile, probability, impressions, clicks, exchange: pyro.sample("purchases", dist.Bernoulli(logits=to_installs(category, creative, ad_type, country, user_profile, probability, impressions, clicks, exchange))), 'purchases'),
    Functor(lambda ad_type, installs, country, clicks, category, device, user_profile, creative,impressions, probability, exchange: pyro.sample("purchases", dist.Bernoulli(logits=to_purchases(ad_type, installs, country, clicks, category, device, user_profile, creative,impressions, probability, exchange))), 'purchases'),
    Functor(lambda budget, purchases, bid_price, country, exchange, category: pyro.sample("revenue", dist.Normal(to_revenue(budget, purchases, bid_price, country, exchange, category), std)), 'revenue'),

}
    , latents)

domain = [BoolDomain('ad_type'), RealDomain('budget', 10, 1100),
          RealDomain('creative', -17000, 100), RealDomain('goal', -1, 74)]

adv_example = SCMOptimizer(adv_scm, domain)
# ------------------------------------------------------------------------------------------------------------------------------------------------
