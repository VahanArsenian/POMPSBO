from copy import deepcopy

from hebo.optimizers.hebo import HEBO, MACE, Mean, Sigma, power_transform, get_model, EvolutionOpt
from hebo.acquisitions.acq import SingleObjectiveAcq, BaseModel, Tensor, Normal
from hebo.acquisitions.acq import EI, LCB
import torch
import pandas as pd
import numpy as np


class CustomEI(SingleObjectiveAcq):

    @property
    def num_obj(self):
        return 1

    def __init__(self, model: BaseModel, best_y, **conf):
        super().__init__(model, **conf)
        self.tau = best_y
        self.eps = conf.get('eps', 1e-4)
        assert (model.num_out == 1)

    def eval(self, x: Tensor, xe: Tensor) -> Tensor:
        with torch.no_grad():
            py, ps2 = self.model.predict(x, xe)
            noise = np.sqrt(2.0) * self.model.noise.sqrt()
            ps = ps2.sqrt().clamp(min=torch.finfo(ps2.dtype).eps)
            # lcb = (py + noise * torch.randn(py.shape)) - self.kappa * ps
            normed = ((self.tau - self.eps - py - noise * torch.randn(py.shape)) / ps)
            dist = torch.distributions.Normal(0., 1.)
            log_phi = dist.log_prob(normed)
            Phi = dist.cdf(normed)
            PI = Phi
            EI = ps * (Phi * normed + log_phi.exp())
            logEIapp = ps.log() - 0.5 * normed ** 2 - (normed ** 2 - 1).log()

            use_app = ~((normed > -6) & torch.isfinite(EI.log()) & torch.isfinite(PI.log())).reshape(-1)
            out = torch.zeros(x.shape[0], self.num_obj)
            # out[:, 0] = lcb.reshape(-1)
            out[:, 0][use_app] = -1 * logEIapp[use_app].reshape(-1)
            out[:, 0][~use_app] = -1 * EI[~use_app].log().reshape(-1)
            return out


class ReducedMACE(MACE):

    @property
    def num_obj(self):
        return 2

    def eval(self, x: torch.FloatTensor, xe: torch.LongTensor) -> torch.FloatTensor:
        """
        minimize (-1 * EI,  -1 * PI, lcb)
        """
        with torch.no_grad():
            py, ps2 = self.model.predict(x, xe)
            noise = np.sqrt(2.0) * self.model.noise.sqrt()
            ps = ps2.sqrt().clamp(min=torch.finfo(ps2.dtype).eps)
            # lcb = (py + noise * torch.randn(py.shape)) - self.kappa * ps
            normed = ((self.tau - self.eps - py - noise * torch.randn(py.shape)) / ps)
            dist = torch.distributions.Normal(0., 1.)
            log_phi = dist.log_prob(normed)
            Phi = dist.cdf(normed)
            PI = Phi
            EI = ps * (Phi * normed + log_phi.exp())
            logEIapp = ps.log() - 0.5 * normed ** 2 - (normed ** 2 - 1).log()
            logPIapp = -0.5 * normed ** 2 - torch.log(-1 * normed) - torch.log(torch.sqrt(torch.tensor(2 * np.pi)))

            use_app = ~((normed > -6) & torch.isfinite(EI.log()) & torch.isfinite(PI.log())).reshape(-1)
            out = torch.zeros(x.shape[0], self.num_obj)
            # out[:, 0] = lcb.reshape(-1)
            out[:, 0][use_app] = -1 * logEIapp[use_app].reshape(-1)
            out[:, 1][use_app] = -1 * logPIapp[use_app].reshape(-1)
            out[:, 0][~use_app] = -1 * EI[~use_app].log().reshape(-1)
            out[:, 1][~use_app] = -1 * PI[~use_app].log().reshape(-1)
            return out


class AdHEBO(HEBO):

    @property
    def model_config(self):
        if self._model_config is None:
            if self.model_name == 'gp':
                cfg = {
                    'lr': 0.01,
                    'num_epochs': 100,
                    'verbose': False,
                    'noise_lb': 8e-4,
                    'pred_likeli': False
                }
            elif self.model_name == 'gpy':
                cfg = {
                    'verbose': False,
                    'warp': False,
                    'space': self.space
                }
            elif self.model_name == 'gpy_mlp':
                cfg = {
                    'verbose': False
                }
            elif self.model_name == 'rf':
                cfg = {
                    'n_estimators': 20
                }
            else:
                cfg = {}
        else:
            cfg = deepcopy(self._model_config)

        if self.space.num_categorical > 0:
            cfg['num_uniqs'] = [len(self.space.paras[name].categories) for name in self.space.enum_names]
        return cfg

    def __init__(self, space, model_name='gpy',

                 rand_sample=None, acq_cls=LCB, es=None, model_config=None):

        super().__init__(space, model_name=model_name,
                         rand_sample=rand_sample, acq_cls=acq_cls, es=es, model_config=model_config)
        print(self.model_config)
        self.__model = None

    def suggest(self, n_suggestions=1, fix_input=None):
        if self.acq_cls != MACE and n_suggestions != 1:
            raise RuntimeError('Parallel optimization is supported only for MACE acquisition')
        if self.X.shape[0] < self.rand_sample:
            sample = self.quasi_sample(n_suggestions, fix_input)
            return sample, None
        else:
            X, Xe = self.space.transform(self.X)
            y = torch.FloatTensor(self.y).clone()
            if self.__model is None:
                model = get_model(self.model_name, self.space.num_numeric, self.space.num_categorical, 1,
                                  **self.model_config)
                # print("fitting GP")
                model.fit(X, Xe, y)
                self.__model = model
            else:
                model = self.__model

            best_id = self.get_best_id(fix_input)
            best_x = self.X.iloc[[best_id]]
            best_y = y.min()
            py_best, ps2_best = model.predict(*self.space.transform(best_x))
            py_best = py_best.detach().numpy().squeeze()
            ps_best = ps2_best.sqrt().detach().numpy().squeeze()

            iter = max(1, self.X.shape[0] // n_suggestions)
            upsi = 0.5
            delta = 0.01
            # kappa = np.sqrt(upsi * 2 * np.log(iter **  (2.0 + self.X.shape[1] / 2.0) * 3 * np.pi**2 / (3 * delta)))
            kappa = np.sqrt(
                upsi * 2 * ((2.0 + self.X.shape[1] / 2.0) * np.log(iter) + np.log(3 * np.pi ** 2 / (3 * delta))))

            acq = self.acq_cls(model, best_y=py_best, kappa=kappa)  # LCB < py_best
            mu = Mean(model)
            sig = Sigma(model, linear_a=-1.)
            opt = EvolutionOpt(self.space, acq, pop=100, iters=100, verbose=False, es=self.es)
            rec = opt.optimize(initial_suggest=best_x, fix_input=fix_input)
            acq_col_name = "__AC_VAL"
            assert len(rec) == len(opt.res.F)

            print(opt.res.F)
            rec['__AC_VAL'] = opt.res.F
            rec = rec[self.check_unique(rec)]

            cnt = 0
            while rec.shape[0] < n_suggestions:
                rand_rec = self.quasi_sample(n_suggestions - rec.shape[0], fix_input)
                rand_rec = rand_rec[self.check_unique(rand_rec)]
                rec = rec.append(rand_rec, ignore_index=True)
                cnt += 1
                # print("Bad entry from", rand_rec)
                if cnt > 3:
                    # sometimes the design space is so small that duplicated sampling is unavoidable
                    break
            if rec.shape[0] < n_suggestions:
                rand_rec = self.quasi_sample(n_suggestions - rec.shape[0], fix_input)
                rec = rec.append(rand_rec, ignore_index=True)

            select_id = np.random.choice(rec.shape[0], n_suggestions, replace=False).tolist()
            # x_guess = []
            # acq_vals = rec[acq_col_name].to_list()
            rec = rec.drop(columns=[acq_col_name])
            x, xe = self.space.transform(rec)
            acq_vals = acq(x, xe).detach().cpu().numpy()
            with torch.no_grad():
                rec_selected = rec.iloc[select_id].copy()
                acq_opt = -np.array(acq_vals)[select_id]
            return rec_selected, acq_opt

    def observe(self, X, y):
        """Feed an observation back.

        Parameters
        ----------
        X : pandas DataFrame
            Places where the objective function has already been evaluated.
            Each suggestion is a dictionary where each key corresponds to a
            parameter being optimized.
        y : array-like, shape (n,1)
            Corresponding values where objective has been evaluated
        """
        valid_id = np.where(np.isfinite(y.reshape(-1)))[0].tolist()
        XX = X.iloc[valid_id]
        yy = y[valid_id].reshape(-1, 1)
        self.X = self.X.append(XX, ignore_index=True)
        self.y = np.vstack([self.y, yy])
        self.__model = None

    def check_unique(self, rec: pd.DataFrame, acq_col="__AC_VAL") -> [bool]:
        return (~pd.concat([self.X, rec[list(set(rec.columns) - {acq_col})]], axis=0).duplicated().tail(
            rec.shape[0]).values).tolist()
