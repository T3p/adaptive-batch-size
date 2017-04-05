import gym
from lqg1d import LQG1D
import numpy as np
import math
import sys

#parallelism
from joblib import Parallel, delayed
import multiprocessing
import tempfile, os

import time

def gauss_policy(s,theta,sigma,noise):
    return s*theta + noise*sigma

def gauss_score(s,a,theta,sigma):
    return (a-s*theta)*s/(sigma**2)

def reinforce_grad(scores,disc_rewards):
    q = np.sum(disc_rewards,1)
    sum_of_scores = np.sum(scores,1)
    #optimal baseline:
    b = np.mean(sum_of_scores**2*q)/np.mean(sum_of_scores**2)
    #gradient estimate:
    return np.mean(sum_of_scores*(q-b))

def gpomdp_grad(scores,disc_rewards):
    H = scores.shape[1]
    cumulative_scores = np.zeros((N,H))
    #optimal baseline:
    b = np.zeros(H)
    for k in range(0,H):
        cumulative_scores[:,k] = sum(scores[:,i] for i in range(0,k+1))
        b[k] = np.mean(cumulative_scores[:,k]**2*disc_rewards[:,k])/ \
                    np.mean(cumulative_scores[:,k]**2)
    #gradient estimate:
    return np.mean(sum(cumulative_scores[:,i]*(disc_rewards[:,i] - b[i]) for i in range(0,H)))
       
def reinforce_d(R,M_phi,H,delta,sigma,gamma):
    return math.sqrt((R**2*M_phi**2*H*(1-gamma**H)**2)/ \
                (sigma**2*(1-gamma)**2*delta)) 

def gpomdp_d(R,M_phi,H,delta,sigma,gamma):
    return math.sqrt((R**2*M_phi**2)/(delta*sigma**2*(1-gamma)**2) * \
                       ((1-gamma**(2*H))/(1-gamma**2)+ H*gamma**(2*H)  - \
                            2 * gamma**H  * (1-gamma**H)/(1-gamma)))  

if __name__ == '__main__':
    env = gym.make('LQG1D-v0')

    theta_star = env.computeOptimalK()[0] 
   
    action_volume = 2*env.max_action #|A|
    R = np.asscalar(env.Q*env.max_pos**2+env.R*env.max_action**2)
    M_phi = env.max_pos

    gamma = env.gamma 
    sigma = 2 #1 
    N = int(sys.argv[1]) #batch size
    H = env.horizon
    theta = 0 #initial value
    delta = 0.2
    grad_estimator = reinforce_grad
    d = reinforce_d(R,M_phi,H,delta,sigma,gamma) #constant for variance bound
    c = (R*M_phi**2*(gamma*math.sqrt(2*math.pi)*sigma + 2*(1-gamma)*action_volume))/ \
            (2*(1-gamma)**3*sigma**3*math.sqrt(2*math.pi))
    epsilon = d/math.sqrt(N)
    
    seed = None
    verbose = 1
    record = len(sys.argv) > 2
    env.seed(seed)
    np.random.seed(seed)  
  
    #trajectory to run in parallel
    def trajectory(n,traces):
        s = env.reset()
        
        #noise realization
        noises = np.random.normal(0,1,H)            

        for l in range(H): 
            a = np.clip(gauss_policy(s,theta,sigma,noises[l]),-2*env.max_action,2*env.max_action)
            traces[n,l,0] = gauss_score(s,a,theta,sigma)
            s,r,_,_ = env.step(a)
            traces[n,l,1] = gamma**l*r 
        
    
    #Learning
    iteration = 0
    path = tempfile.mkdtemp()
    traces_path = os.path.join(path,'traces.mmap')
    n_cores = multiprocessing.cpu_count() 
    while True: 
        iteration+=1 
        if verbose > 0:
            start = time.time()
            print 'iteration:', iteration, 'theta:', theta, 'theta*:', theta_star
            
        #Run N trajectories in parallel  
        traces = np.memmap(traces_path,dtype=float,shape=(N,H,2),mode='w+')  
        Parallel(n_jobs=n_cores)(delayed(trajectory)(n,traces) for n in xrange(N))                  
        scores = traces[:,:,0]
        disc_rewards = traces[:,:,1]
        del traces
        
        #Gradient estimation
        grad_J = grad_estimator(scores,disc_rewards)            

        if verbose > 0:
            print 'epsilon:', epsilon, 'grad:', grad_J
        if verbose > 1:
            N_star = (8/(13-3*math.sqrt(17)))*d**2/grad_J**2
            print 'N*:', N_star  

        #Adaptive step-size
        down = abs(grad_J) - epsilon
        if down<=0:
            break
        up = abs(grad_J) + epsilon
        alpha = down**2/(2*c*up**2)
        if verbose > 0:
            print 'alpha:', alpha
        
        #update
        theta+=alpha*grad_J
        
        if(verbose>0):
            print 'time:', time.time()-start, 's','\n'
    
    print '\nalpha=0 in',iteration,'iterations, theta =',theta
    if record:
        fp = open(sys.argv[2],'a')    

