#envs
import gym
from lqg1d import LQG1D

#parallelism
import gym
from joblib import Paralle,delayed

#performance
import time

#adabatch
from policies import GaussPolicy
from meta_optimization import *
from gradient_estimation import reinforce,gpomdp


#Trajectory (to run in parallel)
def __trajectory(env,tp,states,actions,rewards,n,initial=None,noises=None):
    if noises==None:
        noises = np.random.normal(0,1,tp.H)

    s = env.reset(initial)
    for l in range(tp.H): 
        states[n,l] = s
        a = np.clip(gauss_policy(s,theta,sigma,noises[l]),tp.min_action,tp.max_action)
        actions[n,l] = a
        s,r,_,_ = env.step(a)
        rewards[n,l] = r 

def adabatch(env,tp,pol,delta,con,estimator=gpomdp,bound=bernstein,samp=True,verbose=1):

    N = con.N_min      

    iteration = 0
    while True:
        iteration+=1

        initials = np.random.uniform(tp.min_state,tp.max_state,N)
        states = np.zeros(N,tp.H,pol.feat_dim) 
        actions = np.zeros(N,tp.H,pol.act_dim)
