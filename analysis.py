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
import scipy as sc
import random
import numpy as np

from sklearn.decomposition import PCA
from sklearn.cross_decomposition import CCA
from sklearn.manifold import TSNE
from scipy.stats import linregress
from mdp import MDP
from tqdm import tqdm
from ggplot import * # ggplot style seems better in discrete scatter plot
from common import makedir

TRIAL_SEPARATION = 80
PLOT_LEARNING_CURVE = False
CONFIDENCE_INTERVAL = False
PCA_plot = False
TSNE_plot = True
SOCRE_COMPARE = False
ACTION_COMPARE = True
USE_SELECTED_SUBJECTS = False
SUBJECTS_TO_PLOT = [69, 74, 10, 41, 0] # SWL10 SWL15 KDJ11 HSY22 KDJ1
SUBJECTS_TO_PLOT_LABEL = ['Min', '25 percentile', '50 percentile', '75 percentile', 'Max']
HEAD_AND_TAIL_SUBJECTS = False
FIG_SIZE = (24,14)
SELECTED_EPISODE = [0, 2, 5, 10, 50, 99, 150, 199]
DEFAULT_TITLE = 'Plot-'
ACTION_COLUMN = ['action_' + str(action_num) for action_num in range(MDP.NUM_CONTROL_ACTION)]
HUMAN_DATA_COLUMN = ['MB preference', 'Learning Rate', 'Rel_MF Learning Rate', 'Threshold', 'Inverse Softmax Temp', 'Performance']
DETAIL_COLUMNS = ['rpe', 'spe', 'mf_rel', 'mb_rel', 'p_mb', 'ctrl_reward', 'score', 'action']
COLUMNS = ['rpe', 'spe', 'mf_rel', 'mb_rel', 'p_mb', 'ctrl_reward', 'score'] + ACTION_COLUMN
PCA_COMPONENTS = 2
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

def entropy(sequence):
    p_data  = sequence.value_counts() / len(sequence) # calculates the probabilities
    entropy = sc.stats.entropy(p_data)  # input probabilities to get the entropy 
    return entropy

def save_plt_figure(filename):
    plt.savefig(filename, bbox_inches='tight')
    plt.cla()
    plt.close()

class Analysis:
    """Analysis class
    
    The class consists of two categories of methods
    1. Provide interfaces for collection all kinds of data generated during the simulation
    2. Provide analysis functions of plotting, calculations

    Since plotting functions keep changing and has many details, it is tedious to maintain clear and 
    well-documented plotting functions. So I will not give much explanation to these functions. If you want to
    draw some graphs, it is better to write your own functions instead of trying to understand the old code and modify
    them. Above all, all the data is well-organized and can be accessed easily.

    The following functions are interfaces, they are designed to maintain and update the data this object hold.

    self.data is a dict with key being the mode and a list of data frames being the value
    self.current_data is a list of data frames, being one value in self.data. The length of the list is
    the number of human subjects parameter sets have been used.
    self.current_df is one data frame in self.current_data, its column is COLUMNS

    self.detail, self.current_detail, self.current_detail_df have the same relationship as self.data, self.current_data, self.current_df.
    Except the columns are DETAIL_COLUMNS. And it is trial based, different from self.data episode based. Therefore, using
    self.detail can generate self.data, but it is very slow and inconvenient. Hence it is better to view self.data as a cache of
    self.detail.

    Also, as we can see here, self.current_detail refers to a value in self.detail; self.current_detail_df refers to a value in
    self.current_detail. Hence, they are just iterators in a object scope (set by self.set_current_mode etc.), which is 
    handy to deal with all the data.
    """
    def __init__(self):
        self.data = {}
        self.detail = {}
        self.current_data = []
        self.current_detail = []
        self.trial_separation = TRIAL_SEPARATION
        self.current_df = None
        self.current_detail_df = None
        self.human_data_df = pd.DataFrame(columns=HUMAN_DATA_COLUMN)

    def new_mode(self, mode):
        self.current_data = []
        self.current_detail = []
    
    def save_mode(self, mode):
        self.data[mode] = self.current_data.copy()
        self.detail[mode] = self.current_detail.copy()
        # although human data should be reset in theory, but actually they are the same. So we just let it go

    def set_current_mode(self, mode):
        self.current_data = self.data[mode]
        self.current_detail = self.detail[mode]

    def new_simulation(self):
        makedir(RESULTS_FOLDER) # if created, the function will catch the exception
        self.current_df = pd.DataFrame(columns=COLUMNS)
        self.current_detail_df = pd.DataFrame(columns=DETAIL_COLUMNS)
        datetime_str    = '{:%d-%H-%M-%S}'.format(datetime.datetime.now())
        self.file_name  = (lambda x: RESULTS_FOLDER + x + ' ' + datetime_str)

    def complete_simulation(self):
        self.current_data.append(self.current_df)
        self.current_detail.append(self.current_detail_df)

    def add_detail_res(self, index, res):
        self.current_detail_df.loc[index] = res

    def add_res(self, index, res):
        self.current_df.loc[index] = res

    def add_human_data(self, human_data):
        self.human_data_df.loc[len(self.current_data)] = human_data

    """This is the entry point when using --re-analysis option

    makedir should always be called, but other subsequent calls can be removed depending on the needs.
    """
    def generate_summary(self, title):
        makedir(RESULTS_FOLDER)
        self._compare_score_against_human_data(title)
        self._compare_action_against_human_data(title, 1)
        self._plot_learning_curve(title)

    """All the following functions are plotting functions

    If you are not the author of the function, then you should not worry too much about their actual
    implementation. If you feel hard to understand their logic, it is fine, write your own code based on the
    previous comments on the data structure instead of trying to understand other's code. These code should not mutate
    any existing data.
    """
    def _write_pca_summary(self, pca_obj, f, num_comp=PCA_COMPONENTS):
        f.write('PCA:\n    Explained_Variance_Ratio:\n        ')
        for index in range(num_comp):
            f.write('pc' + str(index) + ': ' + str(pca_obj.explained_variance_ratio_[index]) + ' ')
        f.write('\n    Component:\n')
        for index in range(num_comp):
            f.write('        pc' + str(index) + ':')
            for ratio in pca_obj.components_[index]:
                f.write(' ' + str(ratio))
            f.write('\n')

    def _get_entropy_series(self, episode):
        entropy_series = []
        for detail_df in self.current_detail:
            action_sequence = (detail_df['action'])[episode * self.trial_separation : 
                                                    (episode + 1) * self.trial_separation] # seems exclusive, but loc is inclusive
            entropy_series.append(entropy(action_sequence))
        return entropy_series

    def _scatter_plot(self, analyse_df, x_name, y_name, title, file_name, c_name='entropy', max_val=2, min_val=0, discrete_label=False, num_discrete_val=10):
        if not discrete_label:
            analyse_df.plot(kind='scatter', x=x_name, y=y_name, c=c_name, vmin=min_val, vmax=max_val,
                            title=title, colormap='plasma', marker='s', figsize=FIG_SIZE, alpha=0.8)
            save_plt_figure(file_name)
        else:
            chart = ggplot(analyse_df, aes(x=x_name, y=y_name, color=c_name)) \
                    + geom_point(size=70) \
                    + ggtitle(title) \
                    + theme_bw()
            if num_discrete_val > 6:
                chart += scale_color_brewer(type='qual', palette=3)
            else:
                chart += scale_color_brewer(type='qual', palette="Set1")
            chart.save(file_name)

    def _kl_divergence_against_performance(self, kl_div_list, filename, title):
        KL_DIV = 'KL-divergence'
        SCORE  = 'Negative Log Likelihood Performance'
        FIT_LINE = 'Ordinary Least Square'
        df = pd.DataFrame(columns=[KL_DIV, SCORE])
        df[KL_DIV] = [x[1] for x in kl_div_list]
        df[SCORE]  = self.human_data_df['Performance']
        ax = df.plot(kind='scatter', x=SCORE, y=KL_DIV, marker='x', title=title)
        lowest_10 = df[SCORE].quantile(0.1)
        highest_10 = df[SCORE].quantile(0.9)
        filtered_df = df.loc[(df[SCORE] <= lowest_10) | (df[SCORE] >= highest_10)].copy()
        filtered_df.plot(kind='scatter', x=SCORE, y=KL_DIV, marker='x', c='orange', ax=ax)
        slope, intercept, r_value, p_value, _ = linregress(filtered_df[SCORE], filtered_df[KL_DIV])
        coefficient = (slope, intercept) # linear coefficient
        poly_func = np.poly1d(coefficient)
        filtered_df[FIT_LINE] = [poly_func(score) for score in filtered_df[SCORE]]
        filtered_df.plot(x=SCORE, y=FIT_LINE, c='orange', ax=ax, label="R Squared: {:.3f}\np-value: {:.3f}".format(r_value**2, p_value))
        save_plt_figure(filename)

    def _aggregated_analysis(self, sample_df, feature_series_func, file_name, title, n_pca_comp=1, head_subjects=10, tail_subjects=None, num_sequences=50,
                            feature_label='entropy', in_selected_episodes=True, in_all_episodes=True, simple_analysis=False):
        # PCA
        human_pca  = PCA(n_components=n_pca_comp)
        pca_result = human_pca.fit_transform(sample_df)
        try:
            if PCA_plot:
                with open(file_name('PCA projection' + title), 'x') as f:
                    self._write_pca_summary(human_pca, f, n_pca_comp)
        except FileExistsError:
            pass
        
        # calculate t-SNE with component=2
        sample_tsne = TSNE(n_components=2, learning_rate=200)
        tsne_results = sample_tsne.fit_transform(sample_df)

        # generate just based on given data
        if simple_analysis:
            sample_df_copy = sample_df.copy()
            total_subjects = len(self.current_detail)
            assert num_sequences == sample_df_copy.shape[0] / total_subjects
            feature_index_seq = feature_series_func('dummy_var')
            if USE_SELECTED_SUBJECTS:
                subject_remove_list = SUBJECTS_TO_PLOT + list(filter(lambda x: not x in SUBJECTS_TO_PLOT, list(range(total_subjects))))
                num_subjects  = len(SUBJECTS_TO_PLOT)
                feature_index_seq = [SUBJECTS_TO_PLOT_LABEL[SUBJECTS_TO_PLOT.index(index)] for index in feature_index_seq]
            else:
                kl_divergence = []
                for subject_id in range(total_subjects):
                    sub_tsne = TSNE(n_components=2, perplexity=20)
                    sub_tsne.fit(sample_df_copy.loc[subject_id * num_sequences : 
                                                (subject_id + 1) * num_sequences - 1])
                    kl_divergence.append((subject_id, sub_tsne.kl_divergence_))
                self._kl_divergence_against_performance(kl_divergence, file_name(title + ' Action sequence against Performance'),
                                                       title + ' Action sequence against Performance')
                if tail_subjects is not None:
                    zip_list = []
                    for subject_id in range(total_subjects):
                        zip_list.append((subject_id, feature_index_seq))
                    zip_list.sort(key=lambda pair: pair[1]) # sort by feature sequence because now we have tail subjects to compare
                    subject_remove_list = [x[0] for x in zip_list]
                else:
                    kl_divergence.sort(key=lambda pair: pair[1]) # sort by kl_divergence
                    subject_remove_list = [x[0] for x in kl_divergence]
                tail_cut_index = len(subject_remove_list) if tail_subjects is None else len(subject_remove_list) - tail_subjects
                for subject_to_remove in subject_remove_list[head_subjects:tail_cut_index]:
                    sample_df_copy.drop(sample_df.index[subject_to_remove * num_sequences : (subject_to_remove + 1) * num_sequences], inplace=True)
                    feature_index_seq[subject_to_remove * num_sequences : (subject_to_remove + 1) * num_sequences] = [-1] * num_sequences # mark to remove
                feature_index_seq = list(filter(lambda x: x != -1, feature_index_seq))
            sorted_tsne = TSNE(n_components=2)
            sorted_tsne_res = sorted_tsne.fit_transform(sample_df_copy)
            full_title = 't-SNE ' + title + ' Action Sequence '
            analyse_df = pd.DataFrame()
            analyse_df['t-SNE-1'] = sorted_tsne_res[:,0]
            analyse_df['t-SNE-2'] = sorted_tsne_res[:,1]
            assert analyse_df.shape[0] == (tail_subjects + head_subjects) * num_sequences
            if tail_subjects is not None:
                analyse_df[feature_label] = feature_index_seq # numerical
                analyse_df.plot(kind='scatter', x='t-SNE-1', y='t-SNE-2', c=feature_label,
                                title=full_title, colormap='jet', alpha=0.8)
                save_plt_figure(file_name(full_title))
            else:
                analyse_df[feature_label] = list(map(str, feature_index_seq)) # discrete
                self._scatter_plot(analyse_df, 't-SNE-1', 't-SNE-2', full_title, file_name(full_title), feature_label, discrete_label=True,
                              num_discrete_val=num_subjects)

        # generate feature on selected episodes
        if in_selected_episodes:
            for episode in SELECTED_EPISODE:
                if len(self.current_data[0]) <= episode:
                    print('Episode', episode, 'not found. Skip')
                    continue
                if PCA_plot:
                    full_title = 'PCA_' + title + ' Episode ' + str(episode) + ' Action ' + feature_label
                    analyse_df = pd.DataFrame()
                    analyse_df['PCA-1'] = pca_result[:,0]
                    analyse_df['PCA-2'] = pca_result[:,1]
                    analyse_df[feature_label] = feature_series_func(episode)
                    self._scatter_plot(analyse_df, 'PCA-1', 'PCA-2', full_title, file_name(full_title))
                if TSNE_plot:
                    full_title = 't-SNE_' + title + ' Episode ' + str(episode) + ' Action ' + feature_label
                    analyse_df = pd.DataFrame()
                    analyse_df['t-SNE-1'] = tsne_results[:,0]
                    analyse_df['t-SNE-2'] = tsne_results[:,1]
                    analyse_df[feature_label] = feature_series_func(episode)
                    self._scatter_plot(analyse_df, 't-SNE-1', 't-SNE-2', full_title, file_name(full_title))
            
        if in_all_episodes:
            # calculate t-SNE with component=1
            sample_tsne = TSNE(n_components=1)
            tsne_results = sample_tsne.fit_transform(sample_df)

            # generate feature graph on all episode
            full_title = title + ' Action ' + feature_label
            analyse_df = pd.DataFrame(columns=['PCA-1', feature_label, 'episode', 'MB Preference', 't-SNE-1'])
            for episode in tqdm(range(len(self.current_data[0]))):
                sub_df = pd.DataFrame()
                sub_df['PCA-1'] = pca_result[:,0]
                sub_df[feature_label] = feature_series_func(episode)
                sub_df['episode'] = float(episode)
                sub_df['MB Preference'] = self.human_data_df['MB preference']
                sub_df['t-SNE-1'] = tsne_results[:,0]
                analyse_df = analyse_df.append(sub_df, ignore_index=True)
            if PCA_plot:
                self._scatter_plot(analyse_df, 'episode', 'PCA-1', full_title, file_name('PCA_' + full_title))
            if TSNE_plot:
                self._scatter_plot(analyse_df, 'episode', 't-SNE-1', full_title, file_name('t-SNE_' + full_title))
            self._scatter_plot(analyse_df, 'episode', 'MB Preference', full_title, file_name('MB_' + full_title))

    def _compare_action_against_human_data(self, title, num_comp=PCA_COMPONENTS):
        if not ACTION_COMPARE:
            return
        SAMPLE_ACTION_SEQUENCES   = 50
        NUMBER_OF_SAMPLE_SUBJECTS = 10 if not HEAD_AND_TAIL_SUBJECTS else 9
        NUMBER_OF_TAIL_SUBJECTS   = None if not HEAD_AND_TAIL_SUBJECTS else 9
        makedir(RESULTS_FOLDER + 'Action_Summary/')
        file_name = lambda x: self.file_name('Action_Summary/' + x)
        sample_df = self.human_data_df.copy()
        # self._aggregated_analysis(sample_df, lambda episode: self._get_entropy_series(episode), file_name, title, num_comp)
        sample_df = pd.DataFrame(columns=['trial_' + str(trial_num) for trial_num in range(self.trial_separation)])
        feature_seq = []
        sample_detail_data = self.current_detail # random.sample(self.current_detail, NUMBER_OF_SAMPLE_SUBJECTS)
        for subject_index, detail_df in enumerate(sample_detail_data):
            for index, episode in enumerate(range(len(self.current_data[0]))[-SAMPLE_ACTION_SEQUENCES:]): # extract last 10 episode action sequences
                action_sequence = list(map(int, (detail_df['action'])[episode * self.trial_separation : 
                                                                     (episode + 1) * self.trial_separation].tolist()))
                sample_df.loc[SAMPLE_ACTION_SEQUENCES * subject_index + index] = action_sequence
                if HEAD_AND_TAIL_SUBJECTS:
                    feature_seq.append(self.human_data_df['Performance'].loc[subject_index])
                else:
                    feature_seq.append(subject_index)
        feature_series_func = lambda dummy_var: feature_seq
        self._aggregated_analysis(sample_df, feature_series_func, file_name, title, num_comp, head_subjects=NUMBER_OF_SAMPLE_SUBJECTS,
                                 tail_subjects=NUMBER_OF_TAIL_SUBJECTS, num_sequences=SAMPLE_ACTION_SEQUENCES, 
                                 feature_label='Subject ID' if not HEAD_AND_TAIL_SUBJECTS else 'Negative Log Likelihood Performance', 
                                 in_all_episodes=False, in_selected_episodes=False, simple_analysis=True)
        
    def _compare_score_against_human_data(self, title):
        if not SOCRE_COMPARE:
            return
        makedir(RESULTS_FOLDER + 'Score_Summary/')
        file_name = lambda x: self.file_name('Score_Summary/' + x)
        summary_df = self.human_data_df.copy()
        # create a target for CCA
        target_df  = pd.DataFrame()
        target_df['score'] = [df['score'].mean() for df in self.current_data]
        cca = CCA(n_components=1)
        cca.fit(summary_df, target_df)

        # combine them for PCA
        for column_id in ANALYSIS_EXTRA_COLUMNS:
            summary_df[column_id] = [df[column_id].mean() for df in self.current_data]
        pca = PCA(n_components=PCA_COMPONENTS)
        pca.fit(summary_df)
        with open(file_name('Score Statistics Summary ' + title), 'x') as f:
            self._write_pca_summary(pca, f)
            f.write('\nCCA:\n    X weights:\n')
            f.write('        ' + ' '.join(map(str, cca.x_weights_)))
            f.write('\n    Y weights\n')
            f.write('        ' + ' '.join(map(str, cca.y_weights_)))

        # generate historical CCA
        cca_trace_df = pd.DataFrame(columns=HUMAN_DATA_COLUMN)
        for index in range(self.current_data[0].shape[0])[3:]:
            target_df = pd.DataFrame()
            target_df['score'] = [df['score'].loc[:index].mean() for df in self.current_data]
            cca.fit(self.human_data_df, target_df)
            cca_trace_df.loc[index] = [abs(x[0]) for x in cca.x_weights_]
        cca_trace_df.plot(figsize=FIG_SIZE, grid=True, title='CCA progression summary '+title)
        save_plt_figure(file_name('CCA progression summary ' + title))

    def sequence_to_excel(self, subject_id, column='action'):
        excel_writer = pd.ExcelWriter(self.file_name(column + ' sequence' + ' subject-id ' + str(subject_id)) + '.xlsx')
        sequence_df = pd.DataFrame()
        for mode, subject_list in self.detail.items():
            sequence_df[mode] = (subject_list[subject_id])[column]
        sequence_df.to_excel(excel_writer)
        excel_writer.save()

    def _plot_learning_curve(self, title): # plot smooth learning curve
        if not PLOT_LEARNING_CURVE:
            return
        for index, (data_df, detail_df) in enumerate(zip(self.current_data, self.current_detail)):
            smooth_val = len(data_df) / 50
            smooth_val = int(max(1, smooth_val))
            ctrl_reward_series = data_df['ctrl_reward']
            ma = ctrl_reward_series.rolling(smooth_val).mean()
            plt.figure(figsize=FIG_SIZE)
            plt.plot(ma.index, ma)
            entire_ax = plt.gca()
            plt.xlabel('Episode')
            plt.ylabel('Reward')
            if CONFIDENCE_INTERVAL:
                mstd = ctrl_reward_series.rolling(smooth_val).std()
                plt.title(title + ' Learning curve 95% confidence interval')
                plt.fill_between(mstd.index, ma - 1.96 * mstd, ma + 1.96 * mstd, alpha=0.2)
            else:
                mmin = ctrl_reward_series.rolling(smooth_val).apply(lambda a: np.quantile(a, 0.25), raw=True)
                mmax = ctrl_reward_series.rolling(smooth_val).apply(lambda a: np.quantile(a, 0.75), raw=True)
                plt.title(title + ' Learning curve interquartile range')
                plt.fill_between(mmin.index, mmin, mmax, alpha=0.2)
            # draw a circle around the episode point
            episode_index = 0.95 * len(data_df) # select the episode at 95% position, should be well-trained
            one_episode_df = detail_df.loc[episode_index * self.trial_separation : 
                                          (episode_index + 1) * self.trial_separation - 1].reset_index(drop=True)
            plt.scatter(episode_index, ma[episode_index], s=80, linewidths=3, facecolors='none', edgecolors='orange')
            # plot an episode at well-trained control RL
            episode_plot_axe = plt.axes([.55, .15, .3, .2])
            one_episode_df.plot(y='spe', ax=episode_plot_axe)
            episode_plot_axe.set_xlabel('Trials')
            one_episode_df.plot(y='rpe', ax=episode_plot_axe, secondary_y=True)
            action_series = one_episode_df['action']
            episode_plot_axe.set_xticks(np.arange(0, self.trial_separation, 1))
            # annotate x-axis with action number
            for action_index, action in enumerate(action_series):
                episode_plot_axe.annotate(str(int(action)), xy=((action_index + 1) / (self.trial_separation + 1), 0.05), 
                                          xycoords='axes fraction', annotation_clip=False)
            # annotate on the big picture
            entire_ax.annotate("", xy=(0.7, 0.35), xycoords='figure fraction', 
                xytext=(episode_index, ma[episode_index]), textcoords='data',
                arrowprops=dict(arrowstyle="fancy",
                                color="orange",
                                connectionstyle="arc3,rad=0.3",
                                ),
                )
            save_plt_figure(self.file_name('Learning_curve ' + title + ' parameter set: ' + str(index)))

    def plot_line(self, left_series_names, right_series_names=None, plot_title=None):
        fig, axes = plt.subplots(nrows=2, ncols=1, gridspec_kw = {'height_ratios': [5, 1]})
        if plot_title is None:
            plot_title = DEFAULT_TITLE + str(len(self.current_data) + 1)
        ax1 = self.current_df.loc[:,left_series_names].plot(ax=axes[0], figsize=FIG_SIZE, grid=True, title=plot_title)
        ax1.set_xlabel('Episodes')
        if right_series_names is not None:
            ax2 = self.current_df.loc[:,right_series_names].plot(grid=True, ax=ax1, secondary_y=True)
            ax2.legend(loc=1, bbox_to_anchor=(1.10,0.5))
        ax1.legend(loc=2, bbox_to_anchor=(-0.08,0.5))
        self.human_data_df.loc[len(self.current_data)].plot(kind='bar', ax=axes[1], logy=True)
        save_plt_figure(self.file_name(plot_title))

    def plot_all_human_param(self, title=None):
        self.plot_line(['spe', 'mf_rel', 'mb_rel', 'p_mb'], ['rpe', 'ctrl_reward', 'score'], title)

    def plot_pe(self, mode, title=None):
        self.plot_line(MODE_MAP[mode][0], right_series_names=MODE_MAP[mode][1], plot_title=title)

    def plot_action_effect(self, mode, title=None):
        self.plot_line([MODE_MAP[mode][0]] + ACTION_COLUMN + ['ctrl_reward'], 
                       None if MODE_MAP[mode][1] is None else [MODE_MAP[mode][1]], title)

gData = Analysis()