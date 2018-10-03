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

FIG_SIZE = (24,14)
DEFAULT_TITLE = 'Plot-'
COLUMNS = ['rpe', 'spe', 'mf_rel', 'mb_rel', 'p_mb', 'ctrl_reward']
MODE_MAP = {
    'min-spe' : ('spe'),
    'max-spe' : ('spe'),
    'min-rpe' : ('rpe'),
    'max-rpe' : ('rpe')
}
RESULTS_FOLDER = 'history_results/' + '{:%Y-%m-%d}'.format(datetime.datetime.now()) + '/'
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
        datetime_str = '{:%Y-%m-%d-%H-%M-%S-%f}'.format(datetime.datetime.now())
        plt.savefig(RESULTS_FOLDER + datetime_str, bbox_inches='tight')
        plt.cla()
        plt.close()
        return datetime_str

    def plot_full(self):
        datetime_str = self.plot_line(['spe', 'mf_rel', 'mb_rel', 'p_mb'], ['rpe', 'ctrl_reward'])
        self.current_df.to_msgpack(RESULTS_FOLDER + datetime_str + '.msgpack')

    def plot(self, mode):
        plot_title = mode + ' Agent: ' + str(len(self.data) + 1)
        self.plot_line(MODE_MAP[mode], plot_title=plot_title)

gData = Analysis()

def replot(msgpack_name):
    """Restore previous dataframe and plot again. Can be useful in the future
    to plot with other styles

    Args: 
        msgpack_namemsgpack file name to restore dataframes
    """
    restored_pack = Analysis()
    restored_pack.current_df = pd.read_msgpack(msgpack_name)
    restored_pack.plot_full()