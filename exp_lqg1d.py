import gym
import numpy as np
from lqg1d import LQG1D
import adabatch
from meta_optimization import TaskProp, OptConstr, MetaOptimizer
from policies import GaussPolicy
import utils

#Task
env = gym.make('LQG1D-v0')
tp = TaskProp(
        R = np.asscalar(env.Q*env.max_pos**2+env.R*env.max_action**2),
        M = env.max_pos,
        gamma = 0.9,
        H = 20,
        min_state = -env.max_pos,
        max_state = env.max_pos,
        min_action = -env.max_action,
        max_action = env.max_action,
        volume = 2*env.max_action
)

#Policy
theta_0 = 0
sigma = 1
pol = GaussPolicy(theta_0,sigma**2)

#Features
phi = utils.identity

#Constraints
constr = OptConstr(
            delta=0.95,
            N_min=100,
            N_max=500000,
            N_tot = 30000000
)

if __name__ == '__main__':
    adabatch.learn(env,tp,pol,phi,constr,
        bound_name='bernstein',
        estimator_name='gpomdp',
        emp=True,
        parallel=False
)
