""" Shuangyi Tong <stong@kaist.ac.kr>
    Oct 1, 2018

    Data Analysis Module
    - Data collection
    - Data visualizaton
"""
import datetime
import os
import pandas as pd
import matplotlib.pyplot as plt

from mdp import MDP

FIG_SIZE = (24,14)
DEFAULT_TITLE = 'Plot-'
ACTION_COLUMN = ['action_' + str(action_num) for action_num in range(MDP.NUM_CONTROL_ACTION)]
COLUMNS = ['rpe', 'spe', 'mf_rel', 'mb_rel', 'p_mb', 'ctrl_reward'] + ACTION_COLUMN
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

    def new_simulation(self):
        self.current_df = pd.DataFrame(columns=COLUMNS)
        datetime_str    = '{:%d-%H-%M-%S}'.format(datetime.datetime.now())
        self.file_name  = (lambda x: RESULTS_FOLDER + x + ' ' + datetime_str)

    def complete_simulation(self):
        self.data.append(self.current_df)

    def add_res(self, index, res):
        self.current_df.loc[index] = res

    def plot_line(self, left_series_names, right_series_names=None, plot_title=None):
        if plot_title is None:
            plot_title = DEFAULT_TITLE + str(len(self.data) + 1)
        ax1 = self.current_df.loc[:,left_series_names].plot(figsize=FIG_SIZE, grid=True, title=plot_title)
        ax1.set_xlabel('Episodes')
        if right_series_names is not None:
            ax2 = self.current_df.loc[:,right_series_names].plot(grid=True, ax=ax1, secondary_y=True)
            ax2.legend(loc=1, bbox_to_anchor=(1.10,0.5))
        ax1.legend(loc=2, bbox_to_anchor=(-0.08,0.5))
        plt.savefig(self.file_name(plot_title), bbox_inches='tight')
        plt.cla()
        plt.close()

    def plot_all_human_param(self, title=None):
        self.plot_line(['spe', 'mf_rel', 'mb_rel', 'p_mb'], ['rpe', 'ctrl_reward'], title)
        self.current_df.to_msgpack(self.file_name('RawData') + '.msgpack')

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