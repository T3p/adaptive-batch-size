import gym
from trading_env import TradingEnv
import numpy as np

env = gym.make('trading-v0')

for _ in range(100):
    s = env.reset()
    done = False
    while not done:
        a = np.array([1]) #env.action_space.sample()
        s,r,done,info = env.step(a)
        nav =  info['nav']
        print nav
    print '\n'
