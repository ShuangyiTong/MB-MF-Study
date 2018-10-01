""" Shuangyi Tong <stong@kaist.ac.kr>
    Sept 17, 2018
"""
import torch
import numpy as np

from mdp import MDP
from ddqn import DoubleDQN
from sarsa import SARSA
from forward import FORWARD
from arbitrator import BayesRelEstimator, AssocRelEstimator, Arbitrator

# preset constants
MDP_STAGES           = 2
TOTAL_EPISODES       = 1000
TRIALS_PER_SESSION   = 8
SPE_LOW_THRESHOLD    = 0.1
SPE_HIGH_THRESHOLD   = 0.5
RPE_LOW_THRESHOLD    = 5
RPE_HIGH_THRESHOLD   = 20
CONTROL_REWARD       = 2
CONTROL_REWARD_BIAS  = -10
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
        print("Warning: control mode {0} not found, use default mode {1}".format(mode, DEFAULT_CONTROL_MODE))
        cmp_func = error_reward_map[DEFAULT_CONTROL_MODE]

    if cmp_func(abs(error)):
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
               temperature=Arbitrator.SOFTMAX_TEMPERATURE, rl_learning_rate=SARSA.LEARNING_RATE):
    env     = MDP(MDP_STAGES)
    ddqn    = DoubleDQN(env.observation_space[MDP.CONTROL_AGENT_INDEX],
                        env.action_space[MDP.CONTROL_AGENT_INDEX],
                        torch.cuda.is_available()) # use DDQN for control agent
    sarsa   = SARSA(env.action_space[MDP.HUMAN_AGENT_INDEX], SARSA.RANDOM_PROBABILITY, rl_learning_rate) # SARSA model-free learner
    forward = FORWARD(env.observation_space[MDP.HUMAN_AGENT_INDEX],
                      env.action_space[MDP.HUMAN_AGENT_INDEX],
                      env.state_reward_func, env.output_states_offset, FORWARD.RANDOM_PROBABILITY,
                      FORWARD.TEMPORAL_DISCOUNT_FACTOR, rl_learning_rate) # forward model-based learner
    arb     = Arbitrator(AssocRelEstimator(estimator_learning_rate, MDP.POSSIBLE_OUTPUTS[-1]),
                         BayesRelEstimator(BayesRelEstimator.MEMORY_SIZE, BayesRelEstimator.CATEGORIES, threshold),
                         amp_mb_to_mf, amp_mf_to_mb, temperature)

    # register in the communication controller
    env.agent_comm_controller.register('model-based', forward)

    for episode in range(TOTAL_EPISODES):
        cumulative_p_mb = 0
        cum_mf_rel      = 0
        cum_mb_rel      = 0
        for trials in range(TRIALS_PER_SESSION):
            game_ternimate = False
            human_obs, control_obs_frag = env.reset()
            control_obs = np.append(control_obs_frag, CONTROL_REWARD_BIAS) # use bias to initialize reward
            while not game_ternimate:
                """control agent choose action"""
                control_action = ddqn.action(control_obs)

                """control act on environment"""
                if CTRL_AGENTS_ENABLED:
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
                cumulative_p_mb += p_mb
                cum_mf_rel += mf_rel
                cum_mb_rel += mb_rel

                """update control agent"""
                next_control_obs = np.append(next_control_obs_frag, human_reward)
                ddqn.optimize(control_obs, control_action, next_control_obs, error_to_reward(spe, CONTROL_MODE))
                
                """iterators update"""
                control_obs = next_control_obs
                human_obs   = next_human_obs
        print("p_mb avg: ", (cumulative_p_mb / (TRIALS_PER_SESSION * MDP_STAGES)), end='    ')
        print("mf_rel: ", (cum_mf_rel / (TRIALS_PER_SESSION * MDP_STAGES)), end='   ')
        print("mb_rel: ", (cum_mb_rel / (TRIALS_PER_SESSION * MDP_STAGES)))