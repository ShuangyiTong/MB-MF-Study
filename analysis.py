""" Shuangyi Tong <s9tong@edu.uwaterloo.ca>
    Oct 1, 2018

    Data Analysis Module
    - Data collection
    - Data visualizaton
"""
import datetime
import os
import pandas as pd
import matplotlib.pyplot as plt
import warnings

from sklearn.decomposition import PCA
from sklearn.cross_decomposition import CCA
from mdp import MDP

FIG_SIZE = (24,14)
DEFAULT_TITLE = 'Plot-'
ACTION_COLUMN = ['action_' + str(action_num) for action_num in range(MDP.NUM_CONTROL_ACTION)]
HUMAN_DATA_COLUMN = ['MB preference', 'Learning Rate', 'Rel_MF Learning Rate', 'Threshold', 'Inverse Softmax Temp']
COLUMNS = ['rpe', 'spe', 'mf_rel', 'mb_rel', 'p_mb', 'ctrl_reward', 'score'] + ACTION_COLUMN
PCA_COMPONENTS = 3
ANALYSIS_EXTRA_COLUMNS = ['score', 'rpe', 'spe', 'p_mb']
MODE_MAP = {
    'min-spe' : ['spe', None],
    'max-spe' : ['spe', None],
    'min-rpe' : ['rpe', None],
    'max-rpe' : ['rpe', None],
    'min-rpe-min-spe' : ['spe', 'rpe'],
    'max-rpe-max-spe' : ['spe', 'rpe'],
    'max-rpe-min-spe' : ['spe', 'rpe'],
    'min-rpe-max-spe' : ['spe', 'rpe']
}
RESULTS_FOLDER = 'history_results/' + '{:%Y-%m-%d}'.format(datetime.datetime.now()) + '/' + '{:%Y-%m-%d-%H-%M-%S}'.format(datetime.datetime.now()) + '/'
try:
    os.makedirs(RESULTS_FOLDER)
except:
    pass

class Analysis:
    def __init__(self):
        self.data = []
        self.current_df = None
        self.human_data_df = pd.DataFrame(columns=HUMAN_DATA_COLUMN)

    def new_simulation(self):
        self.current_df = pd.DataFrame(columns=COLUMNS)
        datetime_str    = '{:%d-%H-%M-%S}'.format(datetime.datetime.now())
        self.file_name  = (lambda x: RESULTS_FOLDER + x + ' ' + datetime_str)

    def complete_simulation(self):
        self.data.append(self.current_df)

    def add_res(self, index, res):
        self.current_df.loc[index] = res

    def add_human_data(self, human_data):
        self.human_data_df.loc[len(self.data)] = human_data

    def generate_summary(self, title):
        summary_df = self.human_data_df.copy()
        # create a target for CCA
        target_df  = pd.DataFrame()
        target_df['score'] = [df['score'].mean() for df in self.data]
        cca = CCA(n_components=1)
        cca.fit(summary_df, target_df)

        # combine them for PCA
        for column_id in ANALYSIS_EXTRA_COLUMNS:
            summary_df[column_id] = [df[column_id].mean() for df in self.data]
        pca = PCA(n_components=PCA_COMPONENTS)
        pca.fit(summary_df)
        with open(self.file_name('Statistics Summary '+title), 'x') as f:
            f.write('PCA:\n    Explained_Variance_Ratio:\n        ')
            for index in range(PCA_COMPONENTS):
                f.write('pc' + str(index) + ': ' + str(pca.explained_variance_ratio_[index]) + ' ')
            f.write('\n    Component:\n')
            for index in range(PCA_COMPONENTS):
                f.write('        pc' + str(index) + ':')
                for ratio in pca.components_[index]:
                    f.write(' ' + str(ratio))
                f.write('\n')
            f.write('\nCCA:\n    X weights:\n        ')
            f.write('\n        ' + ' '.join(map(str, cca.x_weights_)))
            f.write('\n    Y weights')
            f.write('\n        ' + ' '.join(map(str, cca.y_weights_)))

        # generate historical CCA
        cca_trace_df = pd.DataFrame(columns=HUMAN_DATA_COLUMN)
        for index in range(self.data[0].shape[0])[3:]:
            target_df = pd.DataFrame()
            target_df['score'] = [df['score'].loc[:index].mean() for df in self.data]
            cca.fit(self.human_data_df, target_df)
            cca_trace_df.loc[index] = [abs(x[0]) for x in cca.x_weights_]
        cca_trace_df.plot(figsize=FIG_SIZE, grid=True, title='CCA progression summary '+title)
        plt.savefig(self.file_name('CCA progression summary '+title), bbox_inches='tight')

    def plot_line(self, left_series_names, right_series_names=None, plot_title=None):
        fig, axes = plt.subplots(nrows=2, ncols=1, gridspec_kw = {'height_ratios': [5, 1]})
        if plot_title is None:
            plot_title = DEFAULT_TITLE + str(len(self.data) + 1)
        ax1 = self.current_df.loc[:,left_series_names].plot(ax=axes[0], figsize=FIG_SIZE, grid=True, title=plot_title)
        ax1.set_xlabel('Episodes')
        if right_series_names is not None:
            ax2 = self.current_df.loc[:,right_series_names].plot(grid=True, ax=ax1, secondary_y=True)
            ax2.legend(loc=1, bbox_to_anchor=(1.10,0.5))
        ax1.legend(loc=2, bbox_to_anchor=(-0.08,0.5))
        self.human_data_df.loc[len(self.data)].plot(kind='bar', ax=axes[1], logy=True)
        plt.savefig(self.file_name(plot_title), bbox_inches='tight')
        plt.cla()
        plt.close()

    def plot_all_human_param(self, title=None):
        self.plot_line(['spe', 'mf_rel', 'mb_rel', 'p_mb'], ['rpe', 'ctrl_reward', 'score'], title)
        self.current_df.to_msgpack(self.file_name('RawData-runtime') + '.msgpack')
        self.human_data_df.to_msgpack(self.file_name('RawData-parameter') + '.msgpack')

    def plot(self, mode, title=None):
        self.plot_line(MODE_MAP[mode][0], right_series_names=MODE_MAP[mode][1], plot_title=title)

    def plot_action_effect(self, mode, title=None):
        self.plot_line(MODE_MAP[mode][0], ACTION_COLUMN + ['ctrl_reward'] if MODE_MAP[mode][1] is None else
                                          [MODE_MAP[mode][1]] + ACTION_COLUMN + ['ctrl_reward'], title)


gData = Analysis()

def replot(msgpack_name, mode='min-spe'):
    """Restore previous dataframe and plot again. Can be useful in the future
    to plot with other styles

    Args: 
        msgpack_namemsgpack file name to restore dataframes
    """
    restored_pack = Analysis()
    restored_pack.new_simulation()
    restored_pack.current_df = pd.read_msgpack(msgpack_name)
    restored_pack.plot_action_effect(mode)