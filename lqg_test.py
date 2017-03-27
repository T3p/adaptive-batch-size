import gym
from ifqi.envs.lqg1d import LQG1D
import time

env = gym.make('LQG1D-v0')

N = 10000
H = 20 
I = 100

for i in range(I):
    print 'iteration:', i
    start = time.time()
    for n in range(N):
        s = env.reset()
        for h in range(H):
            s,r,done,_ = env.step(env.action_space.sample())
            if done:
                env.reset()
    print time.time() - start, 's\n'
