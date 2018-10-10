""" Shuangyi Tong <stong@kaist.ac.kr>
    Sept 17, 2018
"""
import numpy as np
import common

from collections import defaultdict

class FORWARD:
    """FORWARD model-based learner

    See the algorithm description from the publication:
    States versus Rewards: Dissociable Neural Prediction Error Signals Underlying Model-Based
    and Model-Free Reinforcement Learning http://www.princeton.edu/~ndaw/gddo10.pdf

    Currently support Discreate observation and action spaces only
    """
    RANDOM_PROBABILITY       = 0.05
    TEMPORAL_DISCOUNT_FACTOR = 1.0
    LEARNING_RATE            = 0.5
    def __init__(self, observation_space, action_space, state_reward_func, output_offset,
                 epsilon=RANDOM_PROBABILITY, discount_factor=TEMPORAL_DISCOUNT_FACTOR, learning_rate=LEARNING_RATE):
        """Args:
            observation_space (gym.spaces.Discrete)
            action_space (gym.spaces.Discrete)
            state_reward_func (closure): a reward map to initialize state-action value dict
            output_offset (int): specify the starting point of terminal reward state
            epsilon (float): thereshold to make a random action
            learning_rate (float)
            discount_factor (float)
        """
        self.num_states      = observation_space.n
        self.num_actions     = action_space.n
        self.output_offset   = output_offset
        self.epsilon         = epsilon
        self.discount_factor = discount_factor
        self.learning_rate   = learning_rate
        self.T               = {} # transition matrix
        self.reset(state_reward_func)
    
    def _Q_fitting(self):
        """Regenerate state-action value dictionary and put it in a closure

        Return:
            policy_fn (closure)
        """
        self.Q_fwd = defaultdict(lambda: np.zeros(self.num_actions))
        for state in reversed(range(self.num_states)):
            # Do a one-step lookahead to find the best action
            for action in range(self.num_actions):
                for next_state in reversed(range(self.num_states)):
                    prob, reward = self.T[state][action][next_state]
                    if state >= self.output_offset: # terminal reward states at the bottom of the tree
                        reward = 0
                    best_action_value = np.max(self.Q_fwd[next_state])
                    self.Q_fwd[state][action] += prob * (reward + self.discount_factor * best_action_value)

        # Create a deterministic policy using the optimal value function
        self.policy_fn = common.make_epsilon_greedy_policy(self.Q_fwd, self.epsilon, self.num_actions)
        return self.policy_fn

    def action(self, state):
        return self.policy_fn(state)

    def get_Q_values(self, state):
        """Required by some arbitrition processes

        Args:
            state (int): a discrete value representing the state
        
        Return:
            Q_values (list): a list of Q values with indices corresponds to specific action
        """
        return self.Q_fwd[state]

    def optimize(self, state, action, next_state):
        """Optimize state transition matrix
        
        Args:
            state (int)
            action (int)
            next_state (int)
        
        Returns:
            state_prediction_error (float)
        """
        trans_prob = self.T[state][action]
        for post_state in range(self.num_states):
            prob, reward = trans_prob[post_state]
            if post_state == next_state:
                spe = 1 - prob
                trans_prob[post_state] = (prob + self.learning_rate * spe, reward)
            else:
                trans_prob[post_state] = (prob * (1 - self.learning_rate), reward)
        self.T[state][action] = trans_prob
        self._Q_fitting()
        return spe

    def env_reset(self, state_reward_func):
        """Called by the agent communication controller when environment sends a
        reset signal

        Args:
            state_reward_func (closure): as in constructor
        """
        self.reset(state_reward_func, False)

    def reset(self, state_reward_func, reset_trans_prob=True):
        self.state_reward_func = state_reward_func
        for state in range(self.num_states):
            if reset_trans_prob:
                self.T[state] = {action: [] for action in range(self.num_actions)}
            for action in range(self.num_actions):
                for next_state in range(self.num_states):
                    self.T[state][action].append(((1./self.num_states if reset_trans_prob else self.T[state][action][next_state]), 
                                                  self.state_reward_func(next_state)))
        # build state-action value mapping
        self._Q_fitting()
