import os
import pickle
import pandas as pd
import seaborn as sns
import typing as tp
from experiments.pomps_experiment import OptimizationObjective
from pathlib import Path
import matplotlib.pyplot as plt

sns.set_theme()


class Visualizer:

    def __init__(self, root: str, experiment_name: str, objective: OptimizationObjective,
                 max_expected_reward: float, exp_dir: str = None,
                 uncertainty=('pi', 50), central_tendency="median", target: str = "Y"):
        self.root = Path(root)
        self.target = target
        self.uncertainty = uncertainty
        self.central_tendency = central_tendency
        self.objective = objective
        self.experiment_name = experiment_name
        self.exp_dir = experiment_name if exp_dir is None else exp_dir
        self.directory_path = self.root.joinpath(self.exp_dir)
        if not self.directory_path.exists():
            raise FileNotFoundError(self.directory_path)
        self.max_expected_reward = max_expected_reward
        self.files = [f for f in os.listdir(str(self.directory_path)) if f.startswith(self.experiment_name)]
        self.combined_df = pd.concat(list(self.results_iterator()))
        self.policy_freq = self.combined_df[["MPS", "index"]].groupby('index').MPS.value_counts(normalize=True) \
            .reset_index(name='freq')
        self.cum_prob = self.combined_df.sort_values("index")[['EXP_ID', "index", "MPS"]] \
            .pivot(values='MPS', columns='MPS', index=['EXP_ID', 'index']).fillna(
            "-").applymap(lambda x: 0 if x == '-' else 1).groupby(level=[0]).cumsum().reset_index()
        for s in self.combined_df.MPS.unique():
            self.cum_prob[s] = self.cum_prob[s] / (self.cum_prob['index'] + 1)

    def load(self, file_name):
        fn = str(self.directory_path.joinpath(file_name))
        with open(fn, 'rb') as fd:
            return pickle.load(fd)

    def results_iterator(self):
        for idx, f in enumerate(self.files):
            dumps = self.load(f)
            results = dumps['results']
            df = pd.DataFrame(results).reset_index()
            df = df.sort_values("index")
            df['EXP_ID'] = idx
            df['Regret'] = -self.objective.coefficient() * (self.max_expected_reward - df[self.target])
            df['Cum_Regret'] = df['Regret'].cumsum()
            df['Cum_Regret_Norm'] = df['Cum_Regret'] / (df['index']+1)
            yield df

    def plot_pomps_frequency(self, c=None):
        return sns.lineplot(data=self.policy_freq, x='index',
                            y='freq', hue='MPS').set(title="MPS Frequency")

    def plot_target(self, c=None):
        return sns.lineplot(data=self.combined_df, x='index', y=self.target,
                            estimator=self.central_tendency, errorbar=self.uncertainty,
                            label=self.experiment_name).set(title="Target")

    def plot_regret(self, c=None):
        return sns.lineplot(data=self.combined_df, x='index', y='Regret',
                            estimator=self.central_tendency, errorbar=self.uncertainty,
                            label=self.experiment_name).set(title="Regret")

    def plot_cumulative_regret(self, c=None):
        return sns.lineplot(data=self.combined_df, x='index', y='Cum_Regret',
                            estimator=self.central_tendency, errorbar=self.uncertainty,
                            label=self.experiment_name).set(title="Cumulative Regret")

    def plot_cumulative_regret_norm(self, c=None):
        return sns.lineplot(data=self.combined_df, x='index', y='Cum_Regret_Norm',
                            estimator=self.central_tendency, errorbar=self.uncertainty,
                            label=self.experiment_name).set(title="Cumulative Regret / t")

    def plot_cumulative_frequency(self, c=None):
        for s in self.combined_df.MPS.unique():
            sns.lineplot(data=self.cum_prob, x='index', y=s,
                         estimator=self.central_tendency, errorbar=self.uncertainty,
                         label=s).set(title="Cumulative Frequency")

    def _plot(self):
        return [self.plot_pomps_frequency, self.plot_target, self.plot_regret,
                self.plot_cumulative_regret, self.plot_cumulative_regret_norm, self.plot_cumulative_frequency]

    def summary(self):
        for pl in self._plot():
            plt.figure()
            pl()

    @classmethod
    def visualise_multiple(cls, visualisers: tp.List['Visualizer']):
        pal = sns.color_palette()
        for x in zip(*[v._plot() for v in visualisers]):
            plt.figure()
            for idx, xx in enumerate(x):
                xx()
