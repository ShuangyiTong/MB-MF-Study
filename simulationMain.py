""" Shuangyi Tong <stong@kaist.ac.kr>
    Sept 17, 2018
"""
import torch
import getopt
import sys
import numpy as np

from mdp import MDP
from ddqn import DoubleDQN
from sarsa import SARSA
from forward import FORWARD

usage_str = """
Model-free, model-based learning simulation

Usage:
    --mdp-stages [MDP environment stages]         Specify how many stages in MDP environment
    --ctrl-mode <min-spe/max-spe/min-rpe/max-rpe> Choose control agent mode
    --disable-ctrl                                Disable control agents
"""

# preset constants
MDP_STAGES           = 2
TOTAL_EPISODES       = 100
TRIALS_PER_SESSION   = 8
SPE_LOW_THRESHOLD    = 0.3
SPE_HIGH_THRESHOLD   = 0.7
RPE_LOW_THRESHOLD    = 5
RPE_HIGH_THRESHOLD   = 20
CONTROL_REWARD       = 2
CONTROL_REWARD_BIAS  = 0
DEFAULT_CONTROL_MODE = 'max-spe'
CONTROL_MODE         = DEFAULT_CONTROL_MODE
CTRL_AGENTS_ENABLED  = True

error_reward_map = {
    'min-spe' : (lambda x: x < SPE_LOW_THRESHOLD),
    'max-spe' : (lambda x: x > SPE_HIGH_THRESHOLD),
    'min-rpe' : (lambda x: x < RPE_LOW_THRESHOLD),
    'max-rpe' : (lambda x: x > RPE_HIGH_THRESHOLD)
}

def error_to_reward(error, mode=DEFAULT_CONTROL_MODE, bias=CONTROL_REWARD_BIAS):
    try:
        cmp_func = error_reward_map[mode]
    except KeyError:
        print("Warning: control mode {mode} not found, use default mode {DEFAULT_CONTROL_MODE}")
        cmp_func = error_reward_map[DEFAULT_CONTROL_MODE]

    if cmp_func(abs(error)):
        return CONTROL_REWARD + bias
    else:
        return bias

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

def usage():
    print(usage_str)

if __name__ == '__main__':
    short_opt = "h"
    long_opt  = ["help", "mdp-stages=", "disable-control", "ctrl-mode="]
    try:
        opts, args = getopt.getopt(sys.argv[1:], short_opt, long_opt)
    except getopt.GetoptError as err:
        print(err)
        usage()
        sys.exit(2)
    for o, a in opts:
        if o == "--disable-control":
            CTRL_AGENTS_ENABLED = False
        elif o in ("-h", "--help"):
            usage()
            sys.exit()
        elif o == "--mdp-stages":
            MDP_STAGES = a
        elif o == "--ctrl-mode":
            CONTROL_MODE = a
        else:
            assert False, "unhandled option"

    env     = MDP(MDP_STAGES)
    ddqn    = DoubleDQN(env.observation_space[MDP.CONTROL_AGENT_INDEX],
                        env.action_space[MDP.CONTROL_AGENT_INDEX],
                        torch.cuda.is_available()) # use DDQN for control agent
    sarsa   = SARSA(env.action_space[MDP.HUMAN_AGENT_INDEX], SARSA.RANDOM_PROBABILITY) # SARSA model-free learner
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
                ddqn.optimize(control_obs, control_action, next_control_obs, error_to_reward(spe, CONTROL_MODE))
                
                """iterators update"""
                control_obs = next_control_obs
                human_obs   = next_human_obs
        print(cumulative_prediction_error)