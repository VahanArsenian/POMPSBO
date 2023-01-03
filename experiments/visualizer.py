import os
import pickle
import pandas as pd
import seaborn as sns
from experiments.pomps_experiment import OptimizationObjective
from pathlib import Path
import matplotlib.pyplot as plt
sns.set_theme()


class Visualizer:

    def __init__(self, root: str, experiment_name: str, objective: OptimizationObjective,
                 max_expected_reward: float, exp_dir: str = None, uncertainty=('pi', 50), central_tendency="median"):
        self.root = Path(root)
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
        self.policy_freq = self.combined_df[["MPS", "index"]].groupby('index').MPS.value_counts(normalize=True)\
            .reset_index(name='freq')

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
            df['Regret'] = -self.objective.coefficient() * (self.max_expected_reward - df['Y'])
            df['Cum_Regret'] = df['Regret'].cumsum()
            yield df

    def plot_pomps_frequency(self):
        return sns.lineplot(data=self.policy_freq, x='index', y='freq', hue='MPS')

    def plot_target(self, central_tendency='median', uncertainty=('pi', 50)):
        return sns.lineplot(data=self.combined_df, x='index', y='Y', estimator=central_tendency, errorbar=uncertainty)

    def plot_regret(self, central_tendency='median', uncertainty=('pi', 50)):
        return sns.lineplot(data=self.combined_df, x='index', y='Regret',
                            estimator=central_tendency, errorbar=uncertainty)

    def plot_cumulative_regret(self, central_tendency='median', uncertainty=('pi', 50)):
        return sns.lineplot(data=self.combined_df, x='index', y='Cum_Regret',
                            estimator=central_tendency, errorbar=uncertainty)

    def summary(self):
        plt.figure()
        self.plot_pomps_frequency()
        plt.figure()
        self.plot_target(self.central_tendency, self.uncertainty)
        plt.figure()
        self.plot_regret(self.central_tendency, self.uncertainty)
        plt.figure()
        self.plot_cumulative_regret(self.central_tendency, self.uncertainty)

