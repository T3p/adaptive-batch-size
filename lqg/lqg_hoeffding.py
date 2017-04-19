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
    return sum_of_scores*(q-b)

def gpomdp_grad(scores,disc_rewards):
    H = scores.shape[1]
    N = scores.shape[0]
    cumulative_scores = np.zeros((N,H))
    #optimal baseline:
    b = np.zeros(H)
    for k in range(0,H):
        cumulative_scores[:,k] = sum(scores[:,i] for i in range(0,k+1))
        b[k] = np.mean(cumulative_scores[:,k]**2*disc_rewards[:,k])/ \
                    np.mean(cumulative_scores[:,k]**2)
    #gradient estimate:
    return sum(cumulative_scores[:,i]*(disc_rewards[:,i] - b[i]) for i in range(0,H))
       
def reinforce_d(R,M_phi,H,delta,sigma,gamma):
    return math.sqrt((R**2*M_phi**2*H*(1-gamma**H)**2)/ \
                (sigma**2*(1-gamma)**2*delta)) 

def gpomdp_d(R,M_phi,H,delta,sigma,gamma):
    return math.sqrt((R**2*M_phi**2)/(delta*sigma**2*(1-gamma)**2) * \
                       ((1-gamma**(2*H))/(1-gamma**2)+ H*gamma**(2*H)  - \
                            2 * gamma**H  * (1-gamma**H)/(1-gamma)))  

#args: Ninit, delta, output, Nmax
if __name__ == '__main__':
    env = gym.make('LQG1D-v0')

    theta_star = env.computeOptimalK()[0][0] 
   
    action_volume = 2*env.max_action #|A|
    R = np.asscalar(env.Q*env.max_pos**2+env.R*env.max_action**2)
    M_phi = env.max_pos

    gamma = env.gamma 
    sigma = 1 
    N = int(sys.argv[1]) #INITIAL batch size
    H = env.horizon
    theta = 0 #initial value
    delta = float(sys.argv[2])
    grad_estimator = gpomdp_grad
    d = gpomdp_d(R,M_phi,H,delta,sigma,gamma) #constant for variance bound
    c = (R*M_phi**2*(gamma*math.sqrt(2*math.pi)*sigma + 2*(1-gamma)*action_volume))/ \
            (2*(1-gamma)**3*sigma**3*math.sqrt(2*math.pi))  
    seed = None
    verbose = 1
    record = len(sys.argv) > 3
    N_max = np.inf
    if len(sys.argv) > 4:
        N_max = int(sys.argv[4])
    maxGrad = -np.inf
    minGrad = np.inf
 
    #trajectory to run in parallel
    def trajectory(n,initials,noises,traces):
        s = env.reset(initials[n])

        for l in range(H):
            a = np.clip(gauss_policy(s,theta,sigma,noises[n,l]),-env.max_action, env.max_action)
            traces[n,l,0] = gauss_score(s,a,theta,sigma)
            s,r,_,_ = env.step(a)
            traces[n,l,1] = gamma**l*r
    
    if record:
        fp = open(sys.argv[3],'w')    

    #Learning
    J_est = J = -np.inf 
    #Step-size
    alpha = (13-3*math.sqrt(17))/(4*c)
    if verbose>0:
        print 'alpha:', alpha, 'theta*:', theta_star, '\n' 
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
        initials = np.random.uniform(-env.max_pos,env.max_pos,N)
        noises = np.random.normal(0,1,(N,H))
        traces = np.memmap(traces_path,dtype=float,shape=(N,H,2),mode='w+')
        Parallel(n_jobs=n_cores)(delayed(trajectory)(n,initials,noises,traces) for n in xrange(N))  
       
        scores = traces[:,:,0]
        disc_rewards = traces[:,:,1]
        #Performance estimation
        J_est0 = J_est
        J0 = J
        J_est = np.mean(np.sum(disc_rewards,1))
        J = env.computeJ(theta,sigma,N)
        deltaJ_est = J_est - J_est0
        deltaJ = J - J0
        if verbose>0:   
            print 'J:', J, 'J~:', J_est
            print 'deltaJ:', deltaJ, 'deltaJ~:', deltaJ_est
        del traces
        
        #Gradient estimation
        grads = grad_estimator(scores,disc_rewards)
        grad_J = np.mean(grads)
        maxGrad = max(maxGrad,max(grads))
        minGrad = min(minGrad,min(grads))
        if verbose>0:
            print 'hoeffding:', minGrad,maxGrad

        #Stopping condition
        epsilon = math.sqrt((math.log(1/delta)*(maxGrad-minGrad)**2)/(2*N))
        if verbose > 0:
            print 'epsilon:', epsilon, 'grad:', grad_J
        down = abs(grad_J) - epsilon
        if iteration>1 and down<=0:
            break
         
        if record:
            fp.write("{} {} {} {} {} {}\n".format(iteration,N,theta,J,J_est,down))         

        #update
	if iteration>1:
        	theta+=alpha*grad_J

        #Adaptive batch-size (for next batch)
        N = int(math.log(1/delta)*(maxGrad - minGrad)**2*(13+3*math.sqrt(17))/ \
                    (4*grad_J**2)) + 1

        if verbose>0:
            print 'time:', time.time() - start, '\n'
        
        N_tot+=N
        if N_tot>N_max:
            print "Max N reached"
            break
          
    
    print '\nStopped after',iteration,'iterations, theta =',theta
    if record:
        fp.close()

