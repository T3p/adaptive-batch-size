import gym
from ifqi.envs.lqg1d import LQG1D

env = gym.make('LQG1D-v0')
env.reset()
for _ in range(1000):
    env.render()
    env.step(env.action_space.sample()) # take a random action
