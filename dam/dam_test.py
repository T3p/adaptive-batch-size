import gym
import numpy as np
from dam_wrap import DamWrap

env = gym.make('DamWrap-v0')

env.reset()
for _ in xrange(100):
    a = np.random.normal(50,20)
    s,r,_,_ = env.step(a)
    print s
    print r
    print ""
