""" Shuangyi Tong <stong@kaist.ac.kr>
    Sept 17, 2018
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random

from collections import namedtuple
from collections import defaultdict

def make_epsilon_greedy_policy(Q, epsilon, nA):
    """Creates an epsilon-greedy policy based on a given Q-function and epsilon.
    
    Args:
        Q (dict): A dictionary that maps from state -> action-values.
            Each value is a numpy array of length nA (see below)
        epsilon (float): The probability to select a random action. float between 0 and 1.
        nA (int): Number of actions in the environment.
    
    Returns:
        A closure that takes a observation and returns with probability:
            1 - epsilon: action with highest state-action value
            epsilon: random action
    """
    def policy_fn(observation):
        if random.random() < (1 - epsilon):
            return np.argmax(Q[observation])
        else:
            return random.choice(np.arange(nA))

    return policy_fn

class AgentCommController:
    """ Agent Communication Controller: Controls interaction between learning agents
        Designed to be included in a gym.Env class so that all the interactions
        are achieved through gym.Env.step, providing a unified design pattern
        Users may reach this object directly to register the learners

        Learners may include the following functions to be fully functional
        with these classes:
            reset
    """
    def __init__(self):
         # learners are stored in a dict of the form id - list of learners
        self.learners = defaultdict(list)

    def register(self, identifier, learner):
        """Register a learner

        Args:
            identifier: id of the learner
            learner: a learner object satisfies apis shown above
        """
        self.learners[identifier].append(learner)

    def reset(self, identifier, info=None):
        for learner in self.learners[identifier]:
            if info is not None:
                learner.reset(info)
            else:
                learner.reset()

class MLP(nn.Module):
    """Multi Layer Perceptron
    """
    def __init__(self, input_size, output_size):
        """Args:
            input_size (int)
            output_size (int)
        """
        super(MLP, self).__init__() # call parent constructor
        # two fully connected layers
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, output_size)
        
        # parameters initialization
        nn.init.xavier_normal_(self.fc1.weight)
        nn.init.xavier_normal_(self.fc2.weight)
        nn.init.normal_(self.fc1.bias)
        nn.init.normal_(self.fc2.bias)
        
    # need to overload the parent class's forward 
    def forward(self, x):
        x = F.relu((self.fc1(x)))
        x = self.fc2(x)
        return x

    # autograd will do backward for us

class Memory(object):
    """Memory data structure for experience replay, holding 'Event' namedtuple
    Same as deque after Python 3.5
    """
    Event = namedtuple('Event', ['state', 'action', 'next_state', 'reward'])
    def __init__(self, capacity):
        self.capacity = capacity
        self.idx = 0
        self.mem = []
    
    def add_event(self, event):
        if len(self.mem) < self.capacity:
          self.mem.append(event)
        else:
          self.mem[self.idx] = event
        self.idx = (self.idx + 1) % self.capacity
    
    def sample(self, batch_size):
        return random.sample(self.mem, batch_size)
    
    def reset(self):
        self.idx = 0
        self.mem = []
