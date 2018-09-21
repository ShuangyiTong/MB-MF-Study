""" Shuangyi Tong <stong@kaist.ac.kr>
    Sept 17, 2018
"""
import torch
import numpy as np

from mdp import MDP
from ddqn import DoubleDQN
from sarsa import SARSA
from forward import FORWARD

# constants
EPSILON             = 0.05
TOTAL_EPISODES      = 100
TRIALS_PER_SESSION  = 80
SPE_LOW_THRESHOLD   = 0.3
SPE_HIGH_THRESHOLD  = 0.7
RPE_LOW_THRESHOLD   = 5
RPE_HIGH_THRESHOLD  = 20
CONTROL_REWARD      = 1
CONTROL_PUNISHMENT  = -1
CONTROL_REWARD_BIAS = 0
DEFAULT_TASK_MODE   = 'max-spe'

error_reward_map = {
    'min-spe' : (lambda x: x < SPE_LOW_THRESHOLD),
    'max-spe' : (lambda x: x > SPE_HIGH_THRESHOLD),
    'min-rpe' : (lambda x: x < RPE_LOW_THRESHOLD),
    'max-rpe' : (lambda x: x > RPE_HIGH_THRESHOLD)
}

def error_to_reward(error, mode=DEFAULT_TASK_MODE, bias=CONTROL_REWARD_BIAS):
    cmp_func = error_reward_map[mode]
    if cmp_func(abs(error)):
        return CONTROL_REWARD + bias
    else:
        return CONTROL_PUNISHMENT + bias

def arbitrator(model_free_action):
    return model_free_action

def compute_human_action(human_obs, model_free, model_based):
    """Compute human action by compute model-free and model-based separately
    then integrate the result by the arbitrator

    Args:
        human_obs (any): valid in human observation space
        model_free (any callable): model-free agent object
        model_based (any callable): model-based agent object
    
    Return:
        action (int): action to take by human agent
    """
    model_free_action  = model_free.action(human_obs)
    model_based_action = model_based.action(human_obs)
    return arbitrator(model_based_action)

if __name__ == '__main__':
    env     = MDP()
    ddqn    = DoubleDQN(env.observation_space[MDP.CONTROL_AGENT_INDEX],
                        env.action_space[MDP.CONTROL_AGENT_INDEX],
                        torch.cuda.is_available()) # use DDQN for control agent
    sarsa   = SARSA(env.action_space[MDP.HUMAN_AGENT_INDEX], EPSILON) # SARSA model-free learner
    forward = FORWARD(env.observation_space[MDP.HUMAN_AGENT_INDEX],
                      env.action_space[MDP.HUMAN_AGENT_INDEX],
                      env.state_reward_func, env.output_states_offset)

    # register in the communication controller
    env.agent_comm_controller.register('model-based', forward)

    for episode in range(TOTAL_EPISODES):
        cumulative_prediction_error = 0
        for trials in range(TRIALS_PER_SESSION):
            game_ternimate = False
            human_obs, control_obs_frag = env.reset()
            control_obs = np.append(control_obs_frag, CONTROL_REWARD_BIAS) # use bias to initialize reward
            while not game_ternimate:
                """control agent choose action"""
                control_action = ddqn.action(control_obs)

                """control act on environment"""
                _, _, _, _ = env.step([MDP.CONTROL_AGENT_INDEX, control_action])

                """human choose action"""
                human_action = compute_human_action(human_obs, sarsa, forward)

                """human act on environment"""
                next_human_obs, human_reward, game_ternimate, next_control_obs_frag \
                    = env.step((MDP.HUMAN_AGENT_INDEX, human_action))

                """update human agent"""
                spe = forward.optimize(human_obs, human_action, next_human_obs)
                next_human_action = compute_human_action(next_human_obs, sarsa, forward) # required by models like SARSA
                rpe = sarsa.optimize(human_reward, human_action, 
                                     next_human_action, human_obs, next_human_obs)
                cumulative_prediction_error += abs(spe)

                """update control agent"""
                next_control_obs = np.append(next_control_obs_frag, human_reward)
                ddqn.optimize(control_obs, control_action, next_control_obs, error_to_reward(spe))
                
                """iterators update"""
                control_obs = next_control_obs
                human_obs   = next_human_obs
        print(cumulative_prediction_error)