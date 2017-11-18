#envs
import gym
from lqg1d import LQG1D

#parallelism
import sys
from joblib import Parallel,delayed
import multiprocessing
import tempfile,os

#performance
import time

#adabatch
from policies import GaussPolicy
from meta_optimization import *
from gradient_estimation import reinforce,gpomdp,performance


#Trajectory (can be run in parallel)
def __trajectory(env,tp,pol,feature_fun,traces,n,initial=None,noises=None):
    if noises==None:
        noises = np.random.normal(0,1,tp.H)

    s = env.reset(initial)
    for l in range(tp.H): 
        phi = feature_fun(s)
        a = np.clip(pol.act(phi,noises[l]),tp.min_action,tp.max_action)
        s,r,_,_ = env.step(a)
        traces[n,l] = np.concatenate((np.atleast_1d(phi),np.atleast_1d(a),np.atleast_1d(r)))

def adabatch(env,tp,pol,feature_fun,meta_selector,estimator=gpomdp,parallel=True,verbose=1):

    #Initial batch size
    N = con.N_min      

    #Multiprocessing preparation
    if parallel:
        path = tempfile.mkdtemp()
        traces_path = os.path.join(path,'traces.mmap')
        n_cores = multiprocessing.cpu_count()

    #Learning 
    iteration = 0
    while True:
        iteration+=1

        #Collecting experience
        if parallel:
            initials = np.random.uniform(tp.min_state,tp.max_state,N)
            noises = np.random.normal(0,1,(N,tp.H))
            traces = np.memmap(traces_path,dtype=float,shape=(N,tp.H,po.feat_dim+pol.act_dim+1),mode='w+')
            Parallel(n_jobs=n_cores)(delayed(__trajectory)\
                (env,tp,pol,feature_fun,traces,n,initials[n],noises[n],traces) for n in xrange(N))
        else:
            traces = np.zeros((N,tp.H,pol.feat_dim+pol.act_dim+1))
            for n in xrange(N):
                __trajectory(env,tp,pol,feature_fun,traces,n)
        features = traces[:,:,:pol.feat_dim]
        actions = traces[:,:,pol.feat_dim:pol.feat_dim+pol.act_dim]
        rewards = traces[:,:,-1]    

        #Gradient statistics
        grad_samples = estimator(features,actions,rewards,tp.gamma,pol,average=False)
        g_stats = GradStats(grad_samples)

        #Performance statistics 
        J = performance(rewards,gamma) 

        #Meta-optimization
        alpha,N,safe = meta_selector.select(pol,g_stats,tp,N)
        if not safe and verbose:
            print "Unsafe update!"

        #Optimization
        pol.update(alpha*g_stats.get_estimate()) 
