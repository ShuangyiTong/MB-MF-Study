""" Shuangyi Tong <s9tong@edu.uwaterloo.ca>
    Sept 17, 2018
"""
import random
import numpy as np
import gym

from numpy.random import choice
from gym import spaces
from common import AgentCommController

class MDP(gym.Env):
    """Markov Decision Process env class, inherited from gym.Env
    Although there is not much code in gym.Env, it is just
    showing we want to support general gym API, to be able
    to easily run different environment with existing code in the future

    The MDP implemented here support arbitrary stages and arbitrary
    possible actions at any state, but each state share the same number of
    possible actions. So the decision tree is an n-tree
    """

    """MDP constants
    
    Access of observation and action space should refer
    to these indices
    """
    HUMAN_AGENT_INDEX   = 0
    CONTROL_AGENT_INDEX = 1

    LEGACY_MODE           = False
    STAGES                = 2
    TRANSITON_PROBABILITY = [0.9, 0.1]
    NUM_ACTIONS           = 2
    POSSIBLE_OUTPUTS      = [0, 10, 20, 40]
    BIAS_TOGGLE_INCREMENT = 40

    """Control Agent Action Space
    0 - doing nothing
    1 - set stochastic: apply uniform distribution of transition probability
    2 - set deterministic: set transition probability to original one
    3 - randomize human reward: reset reward shape
    4 - reset: randomize human reward and set deterministic transition probability
    """
    NUM_CONTROL_ACTION    = 5

    def __init__(self, stages=STAGES, trans_prob=TRANSITON_PROBABILITY, num_actions=NUM_ACTIONS,
                 outputs=POSSIBLE_OUTPUTS, more_control_input=True, legacy_mode=LEGACY_MODE):
        """
        Args:
            stages (int): stages of the MDP
            trans_prob (list): an array specifying the probability of transitions
            num_actions (int): number of actions possible to take at non-leaf state
                by player. Note total number of possible actions should be multiplied
                by the size of trans_prob
            outputs (list): an array specifying possible outputs
            more_control_input (bool): more element in control observation
            legacy_mode (bool): if use legacy implementation of MDP
        """
        # environment global variables
        self.stages            = stages
        self.human_state       = 0 # start from zero
        self.legacy_mode       = legacy_mode
        if self.legacy_mode:
            self.max_rpe       = outputs[-1] + MDP.BIAS_TOGGLE_INCREMENT
            self.toggle_bit    = 0
        else:
            self.max_rpe       = outputs[-1]

        # human agent variables
        self.action_space      = [spaces.Discrete(num_actions)] # human agent action space
        self.trans_prob        = trans_prob
        self.possible_actions  = len(self.trans_prob) * num_actions
        self.outputs           = outputs # type of outputs
        self.num_output_states = pow(self.possible_actions, self.stages)
        self.output_states = choice(outputs, self.num_output_states)
        self.output_states_offset = int((pow(self.possible_actions, self.stages) - 1)
            / (self.possible_actions - 1)) # geometric series summation
        if self.legacy_mode:
            self.num_states    = self.output_states_offset + len(self.outputs)
        else:
            self.num_states    = self.output_states_offset + self.num_output_states
        self.observation_space = [spaces.Discrete(self.num_states)] # human agent can see states only
        self.state_reward_func = self._make_state_reward_func()

        # control agent variables
        self.more_control_input = more_control_input
        self.action_space.append(spaces.Discrete(MDP.NUM_CONTROL_ACTION)) # control agent action space
        if more_control_input:
            if legacy_mode:
                output_structure = spaces.Discrete(1) # one toggle bit
            else:
                output_structure = spaces.Discrete(self.num_output_states) # output states
            self.observation_space.append(spaces.Tuple((
                output_structure, # depends on if it is legacy mode or 
                spaces.Box(low=0, high=1, shape=(num_actions,), dtype=float), # transition probability 
                spaces.Box(low=0, high=1, shape=(1,), dtype=float), # rpe
                spaces.Box(low=0, high=np.inf, shape=(1,), dtype=float)))) # spe
        else:
            self.observation_space.append(spaces.Tuple((
                spaces.Box(low=0, high=1, shape=(1,), dtype=float), # rpe
                spaces.Box(low=0, high=np.inf, shape=(1,), dtype=float)))) # spe

        # for reset reference
        self.trans_prob_reset = trans_prob

        # agent communication controller
        self.agent_comm_controller = AgentCommController()

    def _make_state_reward_func(self):
        return lambda s: self.output_states[s - self.output_states_offset] \
               if s >= self.output_states_offset else 0

    def _make_control_observation(self):
        if self.more_control_input:
            if self.legacy_mode:
                return np.concatenate([[self.toggle_bit], np.array(self.trans_prob)])
            else:
                return np.concatenate([self.output_states, np.array(self.trans_prob)])
        else:
            return []

    def step(self, action):
        """"Take one step in the environment
        
        Args:
            action ([int, action]): a two element tuple, first sepcify which agent
            second is the action valid in that agent's action space

        Return (human):
            human_obs (int): an integer represent human agent's observation, which is
            equivalent to 'state' in this environment
            human_reward (float): reward received at the end of the game
            done (boolean): if the game termiate
            control_obs_frag (numpy.array): fragment of control observation, need to append reward
        Return (control):
            None, None, None, None: just match the arity
        """
        if action[0] == MDP.HUMAN_AGENT_INDEX:
            """ Human action
            Calculate the index of the n-tree node, start from 0
            Each node has possible_actions childs, the calculation is a little tricky though.
            Output_states_offset is the max index of internal nodes + 1
            Greater or equal to output_states_offset means we need to get reward from output_states
            """
            state = self.human_state * self.possible_actions + \
                    choice(range(action[1] * len(self.trans_prob) + 1, (action[1] + 1) * len(self.trans_prob) + 1),
                           1, True, self.trans_prob)[0]
            reward = self.state_reward_func(state)
            self.human_state = state
            if state < self.output_states_offset:
                done = False
            else:
                done = True
                if self.legacy_mode:
                    self.human_state = self.output_states_offset + self.outputs.index(reward)
                    reward += MDP.BIAS_TOGGLE_INCREMENT if self.toggle_bit else 0
            return self.human_state, reward, done, self._make_control_observation()
        elif action[0] == MDP.CONTROL_AGENT_INDEX:
            """ Control action
            Integrate functional and object oriented programming techniques
            to create this pythonic, compact code, similar to switch in other language
            """
            [lambda env: env, # do nothing
             lambda env: env._set_stochastic_trans_prob(), # uniform trans_prob
             lambda env: setattr(env, 'trans_prob', env.trans_prob_reset), # reset trans prob to deterministic
             lambda env: env._output_reset(), # reset reward shape
             lambda env: env._output_reset_with_deterministic_trans_prob() # reset
            ][action[1]](self)
            return None, None, None, None
        else:
            raise ValueError

    def _set_stochastic_trans_prob(self):
        self.trans_prob = [1./len(self.trans_prob) for i in range(len(self.trans_prob))]

    def _output_swap(self):
        self.output_states = list(map((lambda x: 0 if x == 20 else
                                                 10 if x == 40 else
                                                 20 if x == 0 else
                                                 40), self.output_states))
        self.state_reward_func = self._make_state_reward_func()
        self.agent_comm_controller.reset('model-based', self.state_reward_func)

    def _output_average_with_stochastic_trans_prob(self):
        self.output_states = [0.9 * (x - 20) + 20 for x in self.output_states]
        self.state_reward_func = self._make_state_reward_func()
        self.agent_comm_controller.reset('model-based', self.state_reward_func)
        self._set_stochastic_trans_prob()

    def _output_reset_with_stochastic_trans_prob(self):
        self._output_reset()
        self._set_stochastic_trans_prob()

    def _output_reset_with_deterministic_trans_prob(self):
        self._output_reset()
        self.trans_prob = self.trans_prob_reset

    def _output_reset(self):
        """Reset parameters, used as an action in control agent space
        """
        self.output_states = choice(self.outputs, self.num_output_states)
        # refresh the closure as well
        self.state_reward_func = self._make_state_reward_func()
        # reset human agent
        self.agent_comm_controller.reset('model-based', self.state_reward_func)
        
    def reset(self):
        """Reset the environment before game start or after game terminates

        Return:
            human_obs (int): human agent observation
            control_obs_frag (numpy.array): control agent observation fragment, see step
        """
        self.human_state = 0
        return self.human_state, self._make_control_observation()