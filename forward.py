""" Shuangyi Tong <s9tong@edu.uwaterloo.ca>
    Sept 17, 2018
"""
import numpy as np
import common

from collections import defaultdict
USE_CFORWARD = True
try:
    from lib.cforward import cForward
except ImportError:
    print("Forward C++ dynamic library not found, only pure Python version availiable")
    USE_CFORWARD = False

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
    C_SIZE_TRANSITION_PROB   = 2 # C implementation requries knowing the size of transition probability
    def __init__(self, observation_space, action_space, state_reward_func, output_offset,
                 epsilon=RANDOM_PROBABILITY, discount_factor=TEMPORAL_DISCOUNT_FACTOR, learning_rate=LEARNING_RATE,
                 disable_cforward=False):
        """Args:
            observation_space (gym.spaces.Discrete)
            action_space (gym.spaces.Discrete)
            state_reward_func (closure): a reward map to initialize state-action value dict
            output_offset (int): specify the starting point of terminal reward state
            epsilon (float): thereshold to make a random action
            learning_rate (float)
            discount_factor (float)
        """
        if disable_cforward:
            self.USE_CFORWARD = False
        else:
            self.USE_CFORWARD = USE_CFORWARD
        self.num_states    = observation_space.n
        self.num_actions   = action_space.n
        self.output_offset = output_offset
        if self.USE_CFORWARD:
            self.cforward = cForward(self.num_states, self.num_actions, FORWARD.C_SIZE_TRANSITION_PROB,
                                     self.output_offset, epsilon, learning_rate, discount_factor)
            self.c_reward_array = self.cforward.reward_array
            self.c_Q_buffer     = self.cforward.Q_buf
            self.env_reset(state_reward_func)
        else: # use pure python
            self.epsilon         = epsilon
            self.discount_factor = discount_factor
            self.learning_rate   = learning_rate
            self.T               = {} # transition matrix
            self.reset(state_reward_func)
    
    def _Q_fitting(self):
        """Regenerate state-action value dictionary and put it in a closure

        Return:
            python mode: policy_fn (closure)
            C mode: None
        """
        if self.USE_CFORWARD:
            self.cforward.generate_Q()
        else: # use pure python
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
        if self.USE_CFORWARD:
            raise NotImplementedError("C mode not implemented, switch to pure python to enable action method")
        return self.policy_fn(state)

    def get_Q_values(self, state):
        """Required by some arbitrition processes

        Note if state >= output_state_offset, then
        python mode: the value will be the higest value in all states times a small transition prob
        C mode: 0
        I am not sure why it is implemented like this in python mode (it is moved from legacy code, 
        so I just keep it), but it should make no much difference.

        Args:
            state (int): a discrete value representing the state
        
        Return:
            Q_values (list): a list of Q values with indices corresponds to specific action
        """
        if self.USE_CFORWARD:
            if state < self.output_offset:
                self.cforward.fill_Q_value_buf(int(state))
                Q_values = []
                # need to copy buffer value to real native list, because some method
                # of list may misbehave like len().
                for i in range(self.num_actions):
                    Q_values.append(self.c_Q_buffer[i])
                return Q_values
            else:
                return np.zeros(self.num_actions)
        else: # use pure python
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
        if self.USE_CFORWARD:
            return self.cforward.optimize(int(state), int(action), int(next_state))
        else: # use pure python
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
        if self.USE_CFORWARD:
            # populate reward array
            for i in range(self.num_states - self.output_offset):
                self.c_reward_array[i] = int(state_reward_func(i + self.output_offset))
            self.cforward.generate_Q()
        else: # use pure python
            self.reset(state_reward_func, False)

    def reset(self, state_reward_func, reset_trans_prob=True):
        if self.USE_CFORWARD:
            raise NotImplementedError("full reset method not implemented in C mode")
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
