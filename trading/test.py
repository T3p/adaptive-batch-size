import gym
from trading_env2 import TradingEnv
import numpy as np

env = gym.make('trading-v0')

for _ in range(100):
    s = env._reset(testing=False)
    done = False
    l = 0
    while not done:
        l+=1
        a = np.array([1]) #env.action_space.sample()
        s,r,done,info = env.step(a)
        nav =  info['nav']
        #print s
    #print '\n'
