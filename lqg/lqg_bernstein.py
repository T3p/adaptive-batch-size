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

def reinforce_grads(scores,disc_rewards):
    q = np.sum(disc_rewards,1)
    sum_of_scores = np.sum(scores,1)
    #optimal baseline:
    b = np.mean(sum_of_scores**2*q)/np.mean(sum_of_scores**2)
    #gradient estimate:
    return sum_of_scores*(q-b)

def gpomdp_grads(scores,disc_rewards):
    N = scores.shape[0]
    H = scores.shape[1]
    cumulative_scores = np.zeros((N,H))
    #optimal baseline:
    b = np.zeros(H)
    for k in range(0,H):
        cumulative_scores[:,k] = sum(scores[:,i] for i in range(0,k+1))
        b[k] = np.mean(cumulative_scores[:,k]**2*disc_rewards[:,k])/ \
                    np.mean(cumulative_scores[:,k]**2)
    #gradient estimate:
    return sum(cumulative_scores[:,i]*(disc_rewards[:,i] - b[i]) for i in range(0,H))
       
def cheb_reinforce_d(R,M_phi,H,delta,sigma,gamma):
    return math.sqrt((R**2*M_phi**2*H*(1-gamma**H)**2)/ \
                (sigma**2*(1-gamma)**2*delta)) 

def cheb_gpomdp_d(R,M_phi,H,delta,sigma,gamma):
    return math.sqrt((R**2*M_phi**2)/(delta*sigma**2*(1-gamma)**2) * \
                       ((1-gamma**(2*H))/(1-gamma**2)+ H*gamma**(2*H)  - \
                            2 * gamma**H  * (1-gamma**H)/(1-gamma)))  

def hoeff_d(R,M_phi,H,delta,sigma,gamma,a_max,action_volume,theta):
    assert delta<1
    assert theta<=0
    upper = (a_max - theta*M_phi)*M_phi*action_volume*R/(sigma**2*(1-gamma))
    lower = -upper
    print 'Hoeff upper:', upper
    return math.sqrt((math.log(1/delta)*(upper-lower)**2) \
                        /2)

if __name__ == '__main__':
    env = gym.make('LQG1D-v0')

    theta_star = env.computeOptimalK()[0][0] 
   
    action_volume = 2*env.max_action #|A|
    R = np.asscalar(env.Q*env.max_pos**2+env.R*env.max_action**2)
    M_phi = env.max_pos

    gamma = env.gamma 
    sigma = 1 
    H = env.horizon
    theta = 0 #initial value
    delta = float(sys.argv[1])
    assert delta<1
    grad_estimator = gpomdp_grads
    c = (R*M_phi**2*(gamma*math.sqrt(2*math.pi)*sigma + 2*(1-gamma)*action_volume))/ \
            (2*(1-gamma)**3*sigma**3*math.sqrt(2*math.pi))  
    
    verbose = 1
    record = len(sys.argv) > 2
    seed = None
    np.random.seed(seed)  
    N_max = np.inf
    if len(sys.argv) > 4:
        N_max = int(sys.argv[3])
  
    #trajectory to run in parallel
    def trajectory(n,initial,noises,traces):
        s = env.reset(initial)

        for l in range(H): 
            a = np.clip(gauss_policy(s,theta,sigma,noises[l]),-env.max_action, env.max_action)
            traces[n,l,0] = gauss_score(s,a,theta,sigma)
            s,r,_,_ = env.step(a)
            traces[n,l,1] = gamma**l*r 
        
    if record:
        fp = open(sys.argv[2],'w')    

    #Learning
    J_est = J = -np.inf 
    #Step-size
    #alpha = (13-3*math.sqrt(17))/(4*c)
    if verbose>0:
        print 'theta*:', theta_star, '\n' 
    if record:
        fp.write("{} {} {}\n\n".format(grad_estimator.__name__,delta,theta_star))
    iteration = 0
    path = tempfile.mkdtemp()
    traces_path = os.path.join(path,'traces.mmap')
    n_cores = multiprocessing.cpu_count() 
    N_min = 2
    N = N_min
    N_tot = N
    bad_updates = 0
    while True: 
        iteration+=1 
        if verbose > 0:
            start = time.time()
            print 'iteration:', iteration, 'N:', N, 'theta:', theta  
            
        #Run N trajectories in parallel  
        initials = np.random.uniform(-env.max_pos,env.max_pos,N)
        noises = np.random.normal(0,1,(N,H))
        traces = np.memmap(traces_path,dtype=float,shape=(N,H,2),mode='w+')  
        Parallel(n_jobs=n_cores)(delayed(trajectory)(n,initials[n],noises[n],traces) for n in xrange(N))                  
        scores = traces[:,:,0]
        disc_rewards = traces[:,:,1]
        #Performance estimation
        J_est0 = J_est
        J0 = J
        J_est = np.mean(np.sum(disc_rewards,1))
        J = env.computeJ(theta,sigma,N)
        deltaJ_est = J_est - J_est0
        deltaJ = J - J0
        if iteration>1:
            if deltaJ<0:
                bad_updates+=1
            eff = 1-float(bad_updates)/(iteration-1)
            print 'EFF:', eff, '%'
        
        if verbose>0:   
            print 'J:', J, 'J~:', J_est
            print 'deltaJ:', deltaJ, 'deltaJ~:', deltaJ_est
        del traces
        
        #Gradient estimation
        grad_samples = grad_estimator(scores,disc_rewards)
        grad_J = np.mean(grad_samples)
        sample_var = np.var(grad_samples,ddof=1)            
        d = 7.0/3*math.log(2/delta)
        f = math.sqrt(2*sample_var*math.log(2/delta))/math.sqrt(N_min)
            
        #Adaptive step-size
        actual_eps = d/(N-1) + f
        alpha = (abs(grad_J)-actual_eps)**2/(2*c*(abs(grad_J)+actual_eps)**2) 
        if verbose>0:
                print 'alpha:', alpha

        #Stopping condition
        coeff = [2]
        coeff.append(6*abs(grad_J)+3*d-4*f)
        coeff.append(2*f**2-6*abs(grad_J)*d-12*abs(grad_J)*f-2*d*f)
        coeff.append(-abs(grad_J)*d+6*abs(grad_J)*f**2)
        roots = np.roots(coeff)
        valid_eps = False
        for x in roots:
            if isinstance(x,float) and x>0:
                epsilon = x
                valid_eps = True
                break
        if verbose > 0:
            print 'epsilon:', epsilon, 'grad:', grad_J, 'f:', f
         
        if not valid_eps or epsilon > abs(grad_J):
            epsilon = abs(grad_J)

        if record:
            fp.write("{} {} {} {} {} {}\n".format(iteration,N,theta,alpha,J,J_est))         

        #Update
        theta+=alpha*grad_J

        #Adaptive batch-size (for next batch)
        N = max(N_min,int(d/(epsilon-f) + 1) + 1)  
        
        if verbose>0:
            print 'time:', time.time() - start, '\n'
        
        N_tot+=N
        if N_tot>N_max:
            print "Max N reached"
            break
          
    
    print '\nStopped after',iteration,'iterations, theta =',theta
    if record:
        fp.close()

