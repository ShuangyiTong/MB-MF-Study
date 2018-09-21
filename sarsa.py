""" Shuangyi Tong <stong@kaist.ac.kr>
    Sept 17, 2018
"""
import numpy as np
import common

from collections import defaultdict
from gym import spaces

class SARSA:
    """SARSA model-free learner
    
    Currently support any observation space, but action can only be Discrete
    """
    def __init__(self, action_space, epsilon=0.05, learning_rate=0.2, discount_factor=1):
        """Args:
            action_space (gym.spaces.Discrete)
            epsilon (float): thereshold to make a random action
            learning_rate (float)
            discount_factor (float)
        """
        self.epsilon         = epsilon
        self.num_actions     = action_space.n # assume action space is spaces.Discrete
        self.discount_factor = discount_factor
        self.learning_rate   = learning_rate
        self.reset()
    
    def action(self, state):
        """Act on a given state

        Args:
            state (any): a state valid in the observation space
        
        Returns:
            action (int): a action in a discrete action space
        """
        return self.policy_fn(state)

    def _get_rpe(self, reward, action_taken, next_action, state, next_state):
        return reward + self.discount_factor * self.Q_sarsa[next_state][next_action] \
            - self.Q_sarsa[state][action_taken]

    def optimize(self, reward, action_taken, next_action, state, next_state):
        """Optimize model based on learned experience 

        Args:
            reward (int): reward at next_state
            action_taken (int): action taken in state to get next_state
            next_action (int): next action to be taken
            state (any): representation of sarsa observation
            next_state (any): next sarsa observation after taken action
        
        Return:
            reward_prediction_error (float)
        """
        rpe = self._get_rpe(reward, action_taken, next_action, state, next_state)
        self.Q_sarsa[state][action_taken] += self.learning_rate * rpe
        self.policy_fn = common.make_epsilon_greedy_policy(self.Q_sarsa, self.epsilon, self.num_actions)
        return rpe

    def reset(self):
        self.Q_sarsa   = defaultdict(lambda: np.zeros(self.num_actions)) # default value is a zero numpy array
        self.policy_fn = common.make_epsilon_greedy_policy(self.Q_sarsa, self.epsilon, self.num_actions)