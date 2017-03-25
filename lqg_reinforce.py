import gym
from ifqi.envs.lqg1d import LQG1D
import numpy as np

def gauss_policy(s,theta,sigma):
    mu = theta*s
    return np.random.normal(mu,sigma)

if __name__ == '__main__':
    env = gym.make('LQG1D-v0')
    env.reset()
    theta_star = env.computeOptimalK()[0,0]
    sigma = 0.1

    print('Optimal param: ' + str(theta_star))


    for _ in range(1000):
        env.render()
        a = gauss_policy(env.get_state(),theta_star,sigma)
        env.step(a) 


