#envs
import gym
from lqg1d import LQG1D

#parallelism
import sys
from joblib import Paralle,delayed
import multiprocessing
import tempfile,os

#performance
import time

#adabatch
from policies import GaussPolicy
from meta_optimization import *
from gradient_estimation import reinforce,gpomdp


#Trajectory (to run in parallel)
def __trajectory(env,tp,pol,feature_fun,traces,n,initial=None,noises=None):
    if noises==None:
        noises = np.random.normal(0,1,tp.H)

    s = env.reset(initial)
    for l in range(tp.H): 
        phi = feature_fun(s)
        a = np.clip(pol.act(phi,noises[l]),tp.min_action,tp.max_action)
        s,r,_,_ = env.step(a)
        traces[n,l] = np.concatenate((np.atleast_1d(phi),np.atleast_1d(a),np.atleast_1d(r)))

def adabatch(env,tp,pol,feature_fun,delta,con,estimator=gpomdp,bound=bernstein,samp=True,parallel=True,verbose=1):

    N = con.N_min      

    #Multiprocessing preparation
    if parallel:
        path = tempfile.mkdtemp()
        traces_path = os.path.join(path,'traces.mmap')
        n_cores = multiprocessing.cpu_count()

    iteration = 0
    while True:
        iteration+=1

        if parallel:
            initials = np.random.uniform(tp.min_state,tp.max_state,N)

        noises = np.random.normal(0,1,(N,tp.H))
        traces = np.zeros((N,tp.H,pol.feat_dim+pol.act_dim+1))
        
    
