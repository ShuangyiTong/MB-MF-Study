""" Shuangyi Tong <stong@kaist.ac.kr>
    Sept 17, 2018
"""
import torch
import numpy as np

from tqdm import tqdm
from mdp import MDP
from ddqn import DoubleDQN
from sarsa import SARSA
from forward import FORWARD
from arbitrator import BayesRelEstimator, AssocRelEstimator, Arbitrator
from analysis import gData

# preset constants
MDP_STAGES            = 2
TOTAL_EPISODES        = 1000
TRIALS_PER_SESSION    = 80
SPE_LOW_THRESHOLD     = 0.1
SPE_HIGH_THRESHOLD    = 0.5
RPE_LOW_THRESHOLD     = 5
RPE_HIGH_THRESHOLD    = 20
MIX_LOW_THRESHOLD     = 0.15
MIX_HIGH_THRESHOLD    = 0.7
MF_REL_HIGH_THRESHOLD = 0.8
MF_REL_LOW_THRESHOLD  = 0.5
MB_REL_HIGH_THRESHOLD = 0.7
MB_REL_LOW_THRESHOLD  = 0.3
CONTROL_REWARD        = 20
CONTROL_REWARD_BIAS   = -10
DEFAULT_CONTROL_MODE  = 'max-spe'
CONTROL_MODE          = DEFAULT_CONTROL_MODE
CTRL_AGENTS_ENABLED   = True
RPE_DISCOUNT_FACTOR   = 0.003
ACTION_PERIOD         = 7
STATIC_CONTROL_AGENT  = True

error_reward_map = {
    # x should be a 4-tuple: rpe, spe, mf_rel, mb_rel
    'min-rpe' : (lambda x: x[0] < RPE_LOW_THRESHOLD),
    'max-rpe' : (lambda x: x[0] > RPE_HIGH_THRESHOLD),
    'min-spe' : (lambda x: x[1] < SPE_LOW_THRESHOLD),
    'max-spe' : (lambda x: x[1] > SPE_HIGH_THRESHOLD),
    'min-mf-rel' : (lambda x: x[2] < MF_REL_LOW_THRESHOLD),
    'max-mf-rel' : (lambda x: x[2] > MF_REL_HIGH_THRESHOLD),
    'min-mb-rel' : (lambda x: x[3] < MB_REL_LOW_THRESHOLD),
    'max-mb-rel' : (lambda x: x[3] > MB_REL_HIGH_THRESHOLD),
    'min-rpe-min-spe' : (lambda x: x[0] * RPE_DISCOUNT_FACTOR + x[1] < MIX_LOW_THRESHOLD),
    'max-rpe-max-spe' : (lambda x: x[0] * RPE_DISCOUNT_FACTOR + x[1] > MIX_HIGH_THRESHOLD),
    'min-rpe-max-spe' : (lambda x: x[0] * RPE_DISCOUNT_FACTOR - x[1] < MIX_LOW_THRESHOLD),
    'max-rpe-min-spe' : (lambda x: x[0] * -RPE_DISCOUNT_FACTOR + x[1] < MIX_LOW_THRESHOLD)
}

static_action_map = {
    'min-rpe' : 0,
    'max-rpe' : 3,
    'min-spe' : 0,
    'max-spe' : 1,
    'min-rpe-min-spe' : 0,
    'max-rpe-max-spe' : 4,
    'min-rpe-max-spe' : 2,
    'max-rpe-min-spe' : 3
}

def error_to_reward(error, mode=DEFAULT_CONTROL_MODE, bias=CONTROL_REWARD_BIAS):
    try:
        cmp_func = error_reward_map[mode]
    except KeyError:
        print("Warning: control mode {0} not found, use default mode {1}".format(mode, DEFAULT_CONTROL_MODE))
        cmp_func = error_reward_map[DEFAULT_CONTROL_MODE]

    if cmp_func(error):
        return CONTROL_REWARD + bias
    else:
        return bias

def compute_human_action(arbitrator, human_obs, model_free, model_based):
    """Compute human action by compute model-free and model-based separately
    then integrate the result by the arbitrator

    Args:
        arbitrator (any callable): arbitrator object
        human_obs (any): valid in human observation space
        model_free (any callable): model-free agent object
        model_based (any callable): model-based agent object
    
    Return:
        action (int): action to take by human agent
    """
    return arbitrator.action(model_free.get_Q_values(human_obs),
                             model_based.get_Q_values(human_obs))

def simulation(threshold=BayesRelEstimator.THRESHOLD, estimator_learning_rate=AssocRelEstimator.LEARNING_RATE,
               amp_mb_to_mf=Arbitrator.AMPLITUDE_MB_TO_MF, amp_mf_to_mb=Arbitrator.AMPLITUDE_MF_TO_MB,
               temperature=Arbitrator.SOFTMAX_TEMPERATURE, rl_learning_rate=SARSA.LEARNING_RATE, PARAMETER_SET='DEFAULT'):
    env     = MDP(MDP_STAGES)
    ddqn    = DoubleDQN(env.observation_space[MDP.CONTROL_AGENT_INDEX],
                        env.action_space[MDP.CONTROL_AGENT_INDEX],
                        torch.cuda.is_available()) # use DDQN for control agent
    sarsa   = SARSA(env.action_space[MDP.HUMAN_AGENT_INDEX], learning_rate=rl_learning_rate) # SARSA model-free learner
    forward = FORWARD(env.observation_space[MDP.HUMAN_AGENT_INDEX],
                      env.action_space[MDP.HUMAN_AGENT_INDEX],
                      env.state_reward_func, env.output_states_offset, learning_rate=rl_learning_rate) # forward model-based learner
    arb     = Arbitrator(AssocRelEstimator(estimator_learning_rate, MDP.POSSIBLE_OUTPUTS[-1]),
                         BayesRelEstimator(thereshold=threshold),
                         amp_mb_to_mf=amp_mb_to_mf, amp_mf_to_mb=amp_mf_to_mb)

    # register in the communication controller
    env.agent_comm_controller.register('model-based', forward)

    gData.new_simulation()
    for episode in tqdm(range(TOTAL_EPISODES)):
        cum_p_mb = cum_mf_rel = cum_mb_rel = cum_rpe = cum_spe = cum_ctrl_reward = 0
        cum_ctrl_act = np.zeros(MDP.NUM_CONTROL_ACTION)
        for trials in range(TRIALS_PER_SESSION):
            game_ternimate = False
            human_obs, control_obs_frag = env.reset()
            control_obs = np.append(control_obs_frag, CONTROL_REWARD_BIAS) # use bias to initialize reward
            while not game_ternimate:
                """control agent choose action"""
                if STATIC_CONTROL_AGENT:
                    control_action = static_action_map[CONTROL_MODE]
                else:
                    control_action = ddqn.action(control_obs)
                cum_ctrl_act[control_action] += 1

                """control act on environment"""
                if CTRL_AGENTS_ENABLED and (trials % ACTION_PERIOD is 0 or not STATIC_CONTROL_AGENT):
                    _, _, _, _ = env.step([MDP.CONTROL_AGENT_INDEX, control_action])
                
                """human choose action"""
                human_action = compute_human_action(arb, human_obs, sarsa, forward)

                """human act on environment"""
                next_human_obs, human_reward, game_ternimate, next_control_obs_frag \
                    = env.step((MDP.HUMAN_AGENT_INDEX, human_action))

                """update human agent"""
                spe = forward.optimize(human_obs, human_action, next_human_obs)
                next_human_action = compute_human_action(arb, next_human_obs, sarsa, forward) # required by models like SARSA
                rpe = sarsa.optimize(human_reward, human_action, 
                                     next_human_action, human_obs, next_human_obs)
                mf_rel, mb_rel, p_mb = arb.add_pe(rpe, spe)
                cum_p_mb += p_mb
                cum_mf_rel += mf_rel
                cum_mb_rel += mb_rel
                cum_rpe += abs(rpe)
                cum_spe += spe

                """update control agent"""
                ctrl_reward = error_to_reward((abs(rpe), spe, mf_rel, mb_rel), CONTROL_MODE)
                next_control_obs = np.append(next_control_obs_frag, human_reward)
                ddqn.optimize(control_obs, control_action, next_control_obs, ctrl_reward)
                cum_ctrl_reward += ctrl_reward
                
                """iterators update"""
                control_obs = next_control_obs
                human_obs   = next_human_obs
        total_actions = TRIALS_PER_SESSION * MDP_STAGES
        gData.add_res(episode, list(map(lambda x: x / total_actions,
                                        [cum_rpe, cum_spe, cum_mf_rel, cum_mb_rel, cum_p_mb, cum_ctrl_reward])) + list(cum_ctrl_act))
    gData.plot(CONTROL_MODE, CONTROL_MODE + ' - parameter set: ' + PARAMETER_SET)
    gData.plot_action_effect(CONTROL_MODE, CONTROL_MODE + ' Action Summary - parameter set: ' + PARAMETER_SET)
    gData.plot_all_human_param(CONTROL_MODE + ' Human Agent State - parameter set: ' + PARAMETER_SET)
    gData.complete_simulation()