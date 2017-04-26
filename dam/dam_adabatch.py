import gym
from dam_wrap import DamWrap
import numpy as np
import math
import sys

#parallelism
from joblib import Parallel, delayed
import multiprocessing
import tempfile, os

import time

def gauss_policy(s,theta,sigma,noise):
    return np.dot(s,theta) + noise*sigma

def gauss_score(s,a,theta,sigma):
    return (a-np.dot(s,theta))*s/(sigma**2)

def reinforce_grad(scores,disc_rewards):
    N = scores.shape[0]
    q = np.sum(disc_rewards,1)
    sum_of_scores = np.sum(scores,1) 
    #optimal baseline:
    b = 0
    if N>1:
        b = np.mean(sum_of_scores**2*q)/np.mean(sum_of_scores**2)
    #gradient estimate:
    return np.mean(sum_of_scores*(q-b))

def gpomdp_grad(scores,disc_rewards):
    N = scores.shape[0]
    H = scores.shape[1]
    cumulative_scores = np.zeros((N,H))
    #optimal baseline:
    b = np.zeros(H)
    for k in range(0,H):
        cumulative_scores[:,k] = sum(scores[:,i] for i in range(0,k+1))
        if N>1:
            b[k] = np.mean(cumulative_scores[:,k]**2*disc_rewards[:,k])/ \
                    np.mean(cumulative_scores[:,k]**2)
    #gradient estimate:
    return np.mean(sum(cumulative_scores[:,i]*(disc_rewards[:,i] - b[i]) for i in range(0,H)))

      
def cheb_reinforce_d(R,M_phi,H,delta,sigma,gamma):
    return math.sqrt((R**2*M_phi**2*H*(1-gamma**H)**2)/ \
                (sigma**2*(1-gamma)**2*delta)) 

def cheb_gpomdp_d(R,M_phi,H,delta,sigma,gamma):
    return math.sqrt((R**2*M_phi**2)/(delta*sigma**2*(1-gamma)**2) * \
                       ((1-gamma**(2*H))/(1-gamma**2)+ H*gamma**(2*H)  - \
                            2 * gamma**H  * (1-gamma**H)/(1-gamma)))  

if __name__ == '__main__':
    env = gym.make('DamWrap-v0')
   
    #ENVIRONMENT CONSTANTS
    s_max = 190
    action_volume = env.max_release
    R = 0#0.5*(s_max/env.S - env.H_FLO_U + env.W_IRR)
    M_phi = 1
    m = env.dim

    gamma = 0.99#env.gamma 
    sigma = 10 
    N = int(sys.argv[1]) #INITIAL batch size
    H = env.horizon
    theta = np.zeros(m) #initial value
    theta+=20
    delta = float(sys.argv[2])
    grad_estimator = gpomdp_grad

    verbose = 1
    record = len(sys.argv) > 3
    seed = None
    np.random.seed(seed)  
    N_max = np.inf
    if len(sys.argv) > 4:
        N_max = int(sys.argv[4])
  
    #trajectory to run in parallel
    def trajectory(n,initial,noises,traces):
        s = env.reset(initial)
        
        for l in range(H): 
            a = gauss_policy(s,theta,sigma,noises[l,0])
            traces[n,l,0] =  gauss_score(s,a,theta,sigma) 
            s,r,_,_ = env.step(a,noises[l,1])
            traces[n,l,1,1] = abs(r)
            traces[n,l,1,0] = gamma**l*r 
        
    if record:
        fp = open(sys.argv[3],'w')    

    #Learning
    J_est = J = -np.inf 
    #Step-size
    if record:
        fp.write("{} {} {} {}\n\n".format(grad_estimator.__name__,delta,alpha,theta_star))
    iteration = 0
    path = tempfile.mkdtemp()
    traces_path = os.path.join(path,'traces.mmap')
    n_cores = multiprocessing.cpu_count() 
    N_tot = N
    while True: 
        iteration+=1 
        if verbose > 0:
            start = time.time()
            print 'iteration:', iteration, 'N:', N, 'theta:', theta  
            
        #Run N trajectories in parallel 
        initials = [env.reset(scalar=True) for _ in range(N)]
        noises = np.random.normal(0,1,(N,H,2))
        traces = np.memmap(traces_path,dtype=float,shape=(N,H,2,m),mode='w+')  
        Parallel(n_jobs=n_cores)(delayed(trajectory)(n,initials[n],noises[n],traces) for n in xrange(N))                  
        scores = np.array(traces[:,:,0,:])
        max_seen_reward = max(map(max,traces[:,:,1,1]))
        disc_rewards = np.array(traces[:,:,1,0])
        #Performance estimation
        J_est0 = J_est
        J0 = J
        J_est = np.mean(np.sum(disc_rewards,1))
        J = 0
        deltaJ_est = J_est - J_est0
        deltaJ = J - J0
        if verbose>0:   
            print 'J:', J, 'J~:', J_est
            print 'deltaJ:', deltaJ, 'deltaJ~:', deltaJ_est
        del traces
        
        #Gradient estimation
        grad_J = []
        for j in range(m):
            grad_J.append(grad_estimator(scores[:,:,j],disc_rewards))
        grad_J = np.array(grad_J)            

        #Step-size
        R = max(R,max_seen_reward)
        print 'R:',R
        d = cheb_gpomdp_d(R,M_phi,H,delta,sigma,gamma) #constant for variance bound
        print 'd:',d
        c = (R*M_phi**2*(gamma*math.sqrt(2*math.pi)*sigma + 2*(1-gamma)*action_volume))/ \
            (2*(1-gamma)**3*sigma**3*math.sqrt(2*math.pi))  
        alpha = (13-3*math.sqrt(17))/(4*c)
        if verbose>0:
            print 'alpha:', alpha 

        #Stopping condition
        epsilon = d/math.sqrt(N)
        opt = np.argmax(abs(grad_J))
        gradOpt = grad_J[opt]
        if verbose > 0:
            print 'epsilon:', epsilon, 'grad:', grad_J
        down = abs(gradOpt) - epsilon
        if iteration>1 and down<=0:
            break
 
        if record:
            fp.write("{} {} {} {} {} {}\n".format(iteration,N,theta,J,J_est,down))         

        #update
        if iteration>1:
            theta[opt]+=alpha*gradOpt

        #Adaptive batch-size (for next batch)
        N = int(((13+3*math.sqrt(17))/2)*d**2/gradOpt**2)+1   
        assert N<=10000000

        if verbose>0:
            print 'time:', time.time() - start, '\n'
        
        N_tot+=N
        if N_tot>N_max:
            print "Max N reached"
            break
          
    
    print '\nStopped after',iteration,'iterations, theta =',theta
    if record:
        fp.close()

