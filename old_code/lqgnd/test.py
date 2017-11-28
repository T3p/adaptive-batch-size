import gym
from lqgnd import LQGND

env = gym.make('LQGND-v0')

s = env.reset()

for i in range(100):
    print s
    a = env.action_space.sample()
    print a
    s,r,_,_ = env.step(a)
    print r
    print
