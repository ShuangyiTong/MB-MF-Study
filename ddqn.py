""" Shuangyi Tong <s9tong@edu.uwaterloo.ca>
    Sept 17, 2018
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import gym

from collections import namedtuple
from torch.autograd import Variable
from gym import spaces
from common import MLP, Memory
from functools import reduce

class DoubleDQN:
    """Double DQN model

    Paramters naming and natation are followed from the original paper:
    Deep Reinforcement Learning with Double Q-learning (2015)
    https://arxiv.org/abs/1509.06461
    """
    def __init__(self, observation_space, action_space, use_cuda=False, batch_size=32, 
                 gamma=0.9, tau=50, memory_capacity=1000):
        """Initialize model parameters and training progress variables

        Args:
            observation_space (gym.spaces): a spaces object from gym.spaces module
            action_space (gym.spaces): same as above
            batch_size (int): number of events to be trained in one batch
            gamma (float): discount factor for future rewards
            tau (int): number of episodes delayed before syncing target network
            memory_capacity (int): size of memory
        """
        self.Tensor        = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
        self.LongTensor    = torch.cuda.LongTensor if use_cuda else torch.LongTensor
        self.batch_size    = batch_size
        self.gamma         = gamma
        self.tau           = tau
        self.tau_offset    = 0
        self.replay_memory = Memory(memory_capacity)

        self.input_size    = self._linear_size(observation_space)
        self.output_size   = self._linear_size(action_space)
        self.eval_Q        = MLP(self.input_size, self.output_size) # online network
        self.target_Q      = MLP(self.input_size, self.output_size) # target network
        self.target_Q.load_state_dict(self.eval_Q.state_dict()) # sync target network with online network
        if use_cuda:
            self.eval_Q.cuda()
            self.target_Q.cuda()
        self.optimizer = torch.optim.RMSprop(self.eval_Q.parameters()) # RMSprop for learning eval_Q parameters
        self.criterion = nn.MSELoss() # mean squared error, similar to least squared error

    def _linear_size(self, gym_space):
        """Calculate the size of input/output based on descriptive structure (i.e.
        observation_space/action_space) defined by gym.spaces
        """
        res = 0
        if isinstance(gym_space, spaces.Tuple):
            for space in gym_space.spaces:
                res += self._linear_size(space)
            return res
        elif isinstance(gym_space, spaces.MultiBinary) or \
             isinstance(gym_space, spaces.Discrete):
            return gym_space.n
        elif isinstance(gym_space, spaces.Box):
            return reduce(lambda x,y: x*y, gym_space.shape)
        else:
            raise NotImplementedError

    def action(self, obs):
        with torch.no_grad(): # interence only
            obs_var = Variable(self.Tensor(obs))
            _, action = torch.max(self.eval_Q(obs_var), 0)
        return action.item()

    def optimize(self, obs, action, next_obs, reward):
        """Update memory based on given data
        Train the model if memory capacity reach batch size
        """
        self.replay_memory.add_event(Memory.Event(obs.copy(), action, next_obs.copy(), reward))
        if self.batch_size <= len(self.replay_memory.mem):
            if self.tau == self.tau_offset:
                self.tau_offset = 0
                self.target_Q.load_state_dict(self.eval_Q.state_dict())
            # sample from replay memory
            mini_batch = self.replay_memory.sample(self.batch_size)
            mini_batch = Memory.Event(*zip(*mini_batch)) # do this for batch processing

            # calculate the estimated value
            estimated_value = self.eval_Q(Variable(self.Tensor(mini_batch.state)))
            # select the value associated with the action taken
            estimated_value = estimated_value.gather(1, 
                Variable(self.LongTensor(mini_batch.action).unsqueeze_(1))) # Q(S_t, A_t; theta_t)

            argmax_action = self.eval_Q(Variable(self.Tensor([
                next_state for next_state in mini_batch.next_state if next_state is not None])))
            _, argmax_action = torch.max(argmax_action, 1) # argmax_a Q(S_{t+1}, a; theta_t)

            # calculate target network value
            target_value = self.target_Q(Variable(self.Tensor([
                next_state for next_state in mini_batch.next_state if next_state is not None])))
            target_value = target_value.gather(1, Variable(argmax_action.unsqueeze_(1))) # Q(S_{t+1}, argmax_a Q(S_{t+1}, a; theta_t); theta_t^-)
            target_value *= self.gamma
            target_value += Variable(self.Tensor(mini_batch.reward).unsqueeze_(1)) # R_{t+1}

            # compute the loss between estimated value and target value
            self.optimizer.zero_grad()
            loss = self.criterion(estimated_value, target_value.detach())
            loss.backward() # calculate gradient
            self.optimizer.step() # apply calculated gradient
            
            self.tau_offset += 1