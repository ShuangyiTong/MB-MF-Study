""" Shuangyi Tong <s9tong@edu.uwaterloo.ca>
    Sept 27, 2018
"""

import numpy as np

from math import log, exp
from collections import deque
from random import random

"""Model-free and model-based arbitrator classes

Implemented based on the following publication: 
Neural Computations Underlying Arbitration between Model-Based and Model-free Learning 
http://dx.doi.org/10.1016/j.neuron.2013.11.028
"""

class BayesRelEstimator:
    """Bayesian Reliability Estimation Class
    """
    MEMORY_SIZE     = 1000
    CATEGORIES      = 3
    THRESHOLD       = 0.3
    TARGET_CATEGORY = 0
    DEFAULT_COND_PROB_DIST_FUNC_TEMPLATE = (lambda thereshold: 
        (lambda pe:
            1 if pe < -thereshold else
            0 if pe < thereshold else
            2
        )
    )
    def __init__(self, memory_size=MEMORY_SIZE, categories=CATEGORIES, thereshold=THRESHOLD,
                 cond_prob_dist_func=None, target_category=TARGET_CATEGORY):
        """Args:
            memory_size (int): maximum length of memory, which is the 'T' discrete events
            appeared in the paper
            categories (int): number of categories of prediction errors 
            (negative, zero, positive by default), which is the 'K' parameter in Dirichlet Distribution
            thereshold (float): thereshold for the default three categories, no effect if customized 
            condition probability distribution function provided
            cond_prob_dist_func (closure): a function to separate continuous prediction error
            into discrete categories. the number of categories should match to the categories argument
            If given None, default function will be used
            target_category (int): when calculate reliability, we need to know the target category to
            calculate, in default case it is 0, as appeared on the paper
        
        Construct a rolling container for historic data using deque, use another counter countainer with
        size of categories to cache the number of each category
        """
        self.categories          = categories
        self.pe_records_counter  = np.zeros(self.categories)
        self.pe_records          = deque(maxlen=memory_size)
        self.target_category     = target_category
        self.cond_prob_dist_func = cond_prob_dist_func
        if self.cond_prob_dist_func is None:
            self.cond_prob_dist_func = BayesRelEstimator.DEFAULT_COND_PROB_DIST_FUNC_TEMPLATE(thereshold)

    def add_pe(self, pe, rel_calc=True):
        if len(self.pe_records) == self.pe_records.maxlen:
            self.pe_records_counter[self.pe_records[0]] -= 1
        pe_category = self.cond_prob_dist_func(pe)
        self.pe_records.append(pe_category)
        self.pe_records_counter[pe_category] += 1
        if rel_calc:
            return self.get_reliability()

    def _dirichlet_dist_mean(self, category):
            return (1 + self.pe_records_counter[category]) / \
                   (self.categories + len(self.pe_records))
    
    def _dirichlet_dist_var(self, category):
            return ((1 + self.pe_records_counter[category]) * \
                    (self.categories + len(self.pe_records) - (1 + self.pe_records_counter[category]))) / \
                   (pow((self.categories + len(self.pe_records)), 2) * \
                    (self.categories + len(self.pe_records) + 1))

    def get_reliability(self):
        chi = []
        for category in range(self.categories):
            mean = self._dirichlet_dist_mean(category)
            var  = self._dirichlet_dist_var(category)
            chi.append(mean / var)
        return chi[self.target_category] / sum(chi)

class AssocRelEstimator:
    """Pearce-Hall Associability Reliability Estimation Class
    """
    LEARNING_RATE = 0.2
    MAX_PE        = 40
    def __init__(self, learning_rate=LEARNING_RATE, pe_max=MAX_PE):
        self.chi           = 0
        self.learning_rate = learning_rate
        self.pe_max        = pe_max

    def add_pe(self, pe):
        delta_chi = self.learning_rate * ((1 - abs(pe) / self.pe_max) - self.chi)
        self.chi += delta_chi
        return self.chi

    def get_reliability(self):
        return self.chi

class Arbitrator:
    """Arbitrator class
    """
    AMPLITUDE_MB_TO_MF           = 1
    AMPLITUDE_MF_TO_MB           = 3
    P_MB                         = 0.8
    SOFTMAX_TEMPERATURE          = 0.2
    MAX_TRANSITION_RATE_MF_TO_MB = 1.0
    MAX_TRANSITION_RATE_MB_TO_MF = 1.0
    MF_TO_MB_BOUNDARY_CONDITION  = 0.1
    MB_TO_MF_BOUNDARY_CONDITION  = 0.01 # smaller than mf-to-mb leads to mf control eventually
    def __init__(self, mf_rel_estimator=None, mb_rel_estimator=None, amp_mf_to_mb=AMPLITUDE_MF_TO_MB,
                 amp_mb_to_mf=AMPLITUDE_MB_TO_MF, temperature=SOFTMAX_TEMPERATURE, p_mb=P_MB,
                 max_trans_rate_mf_to_mb=MAX_TRANSITION_RATE_MF_TO_MB, max_trans_rate_mb_to_mf=MAX_TRANSITION_RATE_MB_TO_MF,
                 mf_to_mb_bound=MF_TO_MB_BOUNDARY_CONDITION, mb_to_mf_bound=MB_TO_MF_BOUNDARY_CONDITION):
        self.mf_rel_estimator = mf_rel_estimator if mf_rel_estimator is not None else AssocRelEstimator()
        self.mb_rel_estimator = mb_rel_estimator if mb_rel_estimator is not None else BayesRelEstimator()
        self.A_alpha          = max_trans_rate_mf_to_mb
        self.A_beta           = max_trans_rate_mb_to_mf
        self.B_alpha          = log((1 / mf_to_mb_bound) * self.A_alpha - 1)
        self.B_beta           = log((1 / mb_to_mf_bound) * self.A_beta - 1)
        self.p_mb             = p_mb
        self.p_mf             = self.p_mb
        self.amp_mb_to_mf     = amp_mb_to_mf
        self.amp_mf_to_mb     = amp_mf_to_mb
        self.temperature      = temperature

    def add_pe(self, rpe, spe):
        chi_mf = self.mf_rel_estimator.add_pe(rpe) # reliability of model free 
        chi_mb = self.mb_rel_estimator.add_pe(spe) # reliability of model based
        alpha  = self.A_alpha / (1 + exp(self.B_alpha * chi_mf)) # transition rate MF->MB
        sum_amp= self.amp_mb_to_mf + self.amp_mb_to_mf
        alpha *= self.amp_mf_to_mb / sum_amp # multiply by amplitude
        beta   = self.A_beta  / (1 + exp(self.B_beta * chi_mb)) # transition rate MB->MF
        beta  *= self.amp_mb_to_mf / sum_amp
        self.p_mb += alpha * (1 - self.p_mb) - beta * self.p_mb
        self.p_mf = 1 - self.p_mb
        return chi_mf, chi_mb, self.p_mb

    def action(self, mf_Q_values, mb_Q_values):
        """Choose an action

        Args:
            mf_Q_values (list): a list of size equals to action space size with Q
            values stored in model-free RL agent
            mb_Q_values (list): same as above, but the RL agent in model-based
        
        Calculate integrated Q value
        """
        Q = []
        assert len(mf_Q_values) == len(mb_Q_values), "Q value length not match"
        for action in range(len(mf_Q_values)):
            Q.append(self.p_mf * mf_Q_values[action] + self.p_mb * mf_Q_values[action])

        """Stochastically choose action based on Q value using softmax function
        """
        Q_sum = sum(Q)
        for action in range(len(Q) - 1):
            p_action = exp(self.temperature * Q[action]) / exp(self.temperature * Q_sum)
            if random() < p_action:
                return action
        return len(Q) - 1