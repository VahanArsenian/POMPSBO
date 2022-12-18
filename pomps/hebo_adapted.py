from hebo.optimizers.hebo import HEBO, MACE, Mean, Sigma, power_transform, get_model, EvolutionOpt
import torch
import numpy as np


class AdHEBO(HEBO):

    def __init__(self, space, model_name='gpy',
                 rand_sample=None, acq_cls=MACE, es='nsga2', model_config=None):
        super().__init__(space, model_name=model_name,
                         rand_sample=rand_sample, acq_cls=acq_cls, es=es, model_config=model_config)
        self.__model = None

    def suggest(self, n_suggestions=1, fix_input=None):
        if self.acq_cls != MACE and n_suggestions != 1:
            raise RuntimeError('Parallel optimization is supported only for MACE acquisition')
        if self.X.shape[0] < self.rand_sample:
            sample = self.quasi_sample(n_suggestions, fix_input)
            return sample
        else:
            X, Xe = self.space.transform(self.X)
            try:
                if self.y.min() <= 0:
                    y = torch.FloatTensor(power_transform(self.y / self.y.std(), method='yeo-johnson'))
                else:
                    y = torch.FloatTensor(power_transform(self.y / self.y.std(), method='box-cox'))
                    if y.std() < 0.5:
                        y = torch.FloatTensor(power_transform(self.y / self.y.std(), method='yeo-johnson'))
                if y.std() < 0.5:
                    raise RuntimeError('Power transformation failed')
                if self.__model is None:
                    model = get_model(self.model_name, self.space.num_numeric, self.space.num_categorical, 1,
                                      **self.model_config)
                    print("fitting GP")
                    model.fit(X, Xe, y)
                    self.__model = model
                else:
                    model = self.__model
            except:
                y = torch.FloatTensor(self.y).clone()
                if self.__model is None:
                    model = get_model(self.model_name, self.space.num_numeric, self.space.num_categorical, 1,
                                      **self.model_config)
                    print("fitting GP")
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
            rec = opt.optimize(initial_suggest=best_x, fix_input=fix_input).drop_duplicates()
            print(opt.res.F)
            rec = rec[self.check_unique(rec)]

            cnt = 0
            while rec.shape[0] < n_suggestions:
                rand_rec = self.quasi_sample(n_suggestions - rec.shape[0], fix_input)
                rand_rec = rand_rec[self.check_unique(rand_rec)]
                rec = rec.append(rand_rec, ignore_index=True)
                cnt += 1
                if cnt > 3:
                    # sometimes the design space is so small that duplicated sampling is unavoidable
                    break
            if rec.shape[0] < n_suggestions:
                rand_rec = self.quasi_sample(n_suggestions - rec.shape[0], fix_input)
                rec = rec.append(rand_rec, ignore_index=True)

            select_id = np.random.choice(rec.shape[0], n_suggestions, replace=False).tolist()
            x_guess = []
            with torch.no_grad():
                py_all = mu(*self.space.transform(rec)).squeeze().numpy()
                ps_all = -1 * sig(*self.space.transform(rec)).squeeze().numpy()
                best_pred_id = np.argmin(py_all)
                best_unce_id = np.argmax(ps_all)
                if best_unce_id not in select_id and n_suggestions > 2:
                    select_id[0] = best_unce_id
                if best_pred_id not in select_id and n_suggestions > 2:
                    select_id[1] = best_pred_id
                rec_selected = rec.iloc[select_id].copy()
            return rec_selected

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
