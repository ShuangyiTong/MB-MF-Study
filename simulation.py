""" Shuangyi Tong <s9tong@edu.uwaterloo.ca>
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
TOTAL_EPISODES        = 100
TRIALS_PER_EPISODE    = 80
SPE_LOW_THRESHOLD     = 0.3
SPE_HIGH_THRESHOLD    = 0.5
RPE_LOW_THRESHOLD     = 4
RPE_HIGH_THRESHOLD    = 7
MF_REL_HIGH_THRESHOLD = 0.8
MF_REL_LOW_THRESHOLD  = 0.5
MB_REL_HIGH_THRESHOLD = 0.7
MB_REL_LOW_THRESHOLD  = 0.3
CONTROL_REWARD        = 1
CONTROL_REWARD_BIAS   = 0
INIT_CTRL_INPUT       = [10, 0.5]
DEFAULT_CONTROL_MODE  = 'max-spe'
CONTROL_MODE          = DEFAULT_CONTROL_MODE
CTRL_AGENTS_ENABLED   = True
RPE_DISCOUNT_FACTOR   = 0.003
ACTION_PERIOD         = 3
STATIC_CONTROL_AGENT  = True
ENABLE_PLOT           = True
DISABLE_C_EXTENSION   = False
MORE_CONTROL_INPUT    = True

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
    'min-rpe-min-spe' : lambda x: error_reward_map['min-rpe'](x) and error_reward_map['min-spe'](x),
    'max-rpe-max-spe' : lambda x: error_reward_map['max-rpe'](x) and error_reward_map['max-spe'](x),
    'min-rpe-max-spe' : lambda x: error_reward_map['min-rpe'](x) and error_reward_map['max-spe'](x),
    'max-rpe-min-spe' : lambda x: error_reward_map['max-rpe'](x) and error_reward_map['min-spe'](x)
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
    env     = MDP(MDP_STAGES, more_control_input=MORE_CONTROL_INPUT)
    ddqn    = DoubleDQN(env.observation_space[MDP.CONTROL_AGENT_INDEX],
                        env.action_space[MDP.CONTROL_AGENT_INDEX],
                        torch.cuda.is_available()) # use DDQN for control agent
    arb     = Arbitrator(AssocRelEstimator(estimator_learning_rate, MDP.POSSIBLE_OUTPUTS[-1]),
                         BayesRelEstimator(thereshold=threshold),
                         amp_mb_to_mf=amp_mb_to_mf, amp_mf_to_mb=amp_mf_to_mb)

    gData.new_simulation()
    gData.add_human_data([amp_mf_to_mb / amp_mb_to_mf, rl_learning_rate, estimator_learning_rate, threshold, temperature])
    control_obs_extra = INIT_CTRL_INPUT
    for episode in tqdm(range(TOTAL_EPISODES)):
        # reinitialize human agent every episode
        sarsa   = SARSA(env.action_space[MDP.HUMAN_AGENT_INDEX], learning_rate=rl_learning_rate) # SARSA model-free learner
        forward = FORWARD(env.observation_space[MDP.HUMAN_AGENT_INDEX],
                        env.action_space[MDP.HUMAN_AGENT_INDEX],
                        env.state_reward_func, env.output_states_offset,
                        learning_rate=rl_learning_rate, disable_cforward=DISABLE_C_EXTENSION) # forward model-based learner
        # register in the communication controller
        env.agent_comm_controller.register('model-based', forward)
        cum_p_mb = cum_mf_rel = cum_mb_rel = cum_rpe = cum_spe = cum_reward = cum_score = 0
        cum_ctrl_act = np.zeros(MDP.NUM_CONTROL_ACTION)
        for trial in range(TRIALS_PER_EPISODE):
            t_p_mb = t_mf_rel = t_mb_rel = t_rpe = t_spe = t_reward = t_score = 0
            game_terminate              = False
            human_obs, control_obs_frag = env.reset()
            control_obs                 = np.append(control_obs_frag, control_obs_extra)

            """control agent choose action"""
            if STATIC_CONTROL_AGENT:
                control_action = static_action_map[CONTROL_MODE]
            else:
                control_action = ddqn.action(control_obs)
            cum_ctrl_act[control_action] += 1

            """control act on environment"""
            if CTRL_AGENTS_ENABLED and (trial % ACTION_PERIOD is 0 or not STATIC_CONTROL_AGENT):
                _, _, _, _ = env.step([MDP.CONTROL_AGENT_INDEX, control_action])

            while not game_terminate:
                """human choose action"""
                human_action = compute_human_action(arb, human_obs, sarsa, forward)

                """human act on environment"""
                next_human_obs, human_reward, game_terminate, next_control_obs_frag \
                    = env.step((MDP.HUMAN_AGENT_INDEX, human_action))

                """update human agent"""
                spe = forward.optimize(human_obs, human_action, next_human_obs)
                next_human_action = compute_human_action(arb, next_human_obs, sarsa, forward) # required by models like SARSA
                rpe = sarsa.optimize(human_reward, human_action, 
                                    next_human_action, human_obs, next_human_obs)
                mf_rel, mb_rel, p_mb = arb.add_pe(rpe, spe)
                t_p_mb   += p_mb
                t_mf_rel += mf_rel
                t_mb_rel += mb_rel
                t_rpe    += abs(rpe)
                t_spe    += spe
                t_score  += human_reward # if not the terminal state, human_reward is 0, so simply add here is fine
                
                """iterators update"""
                human_obs = next_human_obs
            
            # calculation after one trial
            p_mb, mf_rel, mb_rel, rpe, spe = list(map(lambda x: x / MDP_STAGES, [
            t_p_mb, t_mf_rel, t_mb_rel, t_rpe, t_spe])) # map to average value

            cum_p_mb   += p_mb
            cum_mf_rel += mf_rel
            cum_mb_rel += mb_rel
            cum_rpe    += rpe
            cum_spe    += spe
            cum_score  += t_score

            """update control agent"""
            t_reward = error_to_reward((rpe, spe, mf_rel, mb_rel), CONTROL_MODE)
            cum_reward += t_reward
            next_control_obs = np.append(next_control_obs_frag, [rpe, spe])
            ddqn.optimize(control_obs, control_action, next_control_obs, t_reward)
            control_obs_extra = [rpe, spe]
            gData.add_detail_res(trial + TRIALS_PER_EPISODE * episode, 
                                [rpe, spe, mf_rel, mb_rel, p_mb, t_reward, t_score] + [control_action])
        gData.add_res(episode, 
                      list(map(lambda x: x / TRIALS_PER_EPISODE, 
                               [cum_rpe, cum_spe, cum_mf_rel, cum_mb_rel, cum_p_mb, cum_reward, cum_score] + 
                               list(cum_ctrl_act))))

    if ENABLE_PLOT:
        gData.plot_all_human_param(CONTROL_MODE + ' Human Agent State - parameter set: ' + PARAMETER_SET)
        gData.plot_pe(CONTROL_MODE, CONTROL_MODE + ' - parameter set: ' + PARAMETER_SET)
        gData.plot_action_effect(CONTROL_MODE, CONTROL_MODE + ' Action Summary - parameter set: ' + PARAMETER_SET)
    gData.complete_simulation()