"""Simple RL task to verify programming correctness"""

import gym
import numpy as np

from ddqn import DoubleDQN

env = gym.make('CartPole-v1')
agent = DoubleDQN(env.observation_space, env.action_space)
EPISODES = 1000
for e in range(EPISODES):
    state = env.reset()
    for time in range(500):
        env.render()
        action = agent.action(state)
        next_state, reward, done, _ = env.step(action)
        reward = reward if not done else -100
        agent.optimize(state, action, next_state, reward)
        state = next_state
        if done:
            print("episode: {}/{}, score: {}"
                    .format(e, EPISODES, time))
            break
env.close()