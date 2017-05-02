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

def reinforce(scores,disc_rewards):
    q = np.sum(disc_rewards,1)
    sum_of_scores = np.sum(scores,1)
    #optimal baseline:
    b = np.mean(sum_of_scores**2*q)/np.mean(sum_of_scores**2)
    #gradient estimates:
    return sum_of_scores*(q-b)

def gpomdp(scores,disc_rewards):
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

#Estimation error corresponding to optimal batch-size
def eps_opt(d,f,grad_J):
    return 1.0/4*(math.sqrt(17*grad_J**2 + 18*abs(grad_J)*f + f**2) - 3*abs(grad_J) + f)
    
def cheb_reinforce(R,M_phi,H,delta,sigma,gamma,grad_range=None,sample_var=None,grad_J=None):
    d =  math.sqrt((R**2*M_phi**2*H*(1-gamma**H)**2)/ \
                (sigma**2*(1-gamma)**2*delta)) 
    return d,0,0

def cheb_gpomdp(R,M_phi,H,delta,sigma,gamma,grad_range=None,sample_var=None,grad_J=None):
    d = math.sqrt((R**2*M_phi**2)/(delta*sigma**2*(1-gamma)**2) * \
                       ((1-gamma**(2*H))/(1-gamma**2)+ H*gamma**(2*H)  - \
                            2 * gamma**H  * (1-gamma**H)/(1-gamma)))
    return d,0,0

#N.B. Range valid only in case of non-negative or non-positive reward
def grad_range(R,M_phi,sigma,gamma,a_max,action_volume,theta):
    Q_sup = action_volume*R/(1-gamma)
    return 2*M_phi*a_max/sigma**2*Q_sup

def hoeffding(R,M_phi,H,delta,sigma,gamma,grad_range,sample_var=None,grad_J=None):
    assert delta<1
    d = grad_range*math.sqrt(math.log(2/delta)/2)
    return d,0,0

def bernstein(R,M_phi,H,delta,sigma,gamma,grad_range,sample_var,grad_J=None):
    assert delta < 1
    d = math.sqrt(2*sample_var*math.log(3/delta))
    f = 3*math.log(3/delta)*grad_range/N_min
    return d,f,f*N_min

def bernstein2(R,M_phi,H,delta,sigma,gamma,grad_range,sample_var,grad_J=None):
    assert delta < 1
    d = math.sqrt(2*sample_var*math.log(3/delta))
    f0 = 3*math.log(3/delta)*grad_range
    return d+f0,0,0

def iter_bernstein(R,M_phi,H,delta,sigma,gamma,grad_range,sample_var,grad_J):
    d = math.sqrt(2*sample_var*math.log(3/delta))
    f0 = 3*math.log(3/delta)*grad_range
    N_inf = N_min
    for n in xrange(N_min,N_max+1):
        f = f0/n
        eps = eps_opt(d,f,grad_J)
        N_opt = int(((d + math.sqrt(d**2 + 4*eps*f0))/(2*eps))**2) + 1
        if N_opt >= n:
            N_inf = n
        else:
            break
    print 'Nmin:', N_inf
    f = f0/N_inf
    return d,f,f0

if __name__ == '__main__':
    env = gym.make('LQG1D-v0')

    theta_star = env.computeOptimalK()[0][0] 
   
    a_max = env.max_action
    action_volume = 2*a_max  #|A|
    R = np.asscalar(env.Q*env.max_pos**2+env.R*env.max_action**2)
    M_phi = env.max_pos

    gamma = env.gamma 
    sigma = 1 
    H = env.horizon
    theta = 0 #initial value

    estimators = [reinforce,gpomdp]
    bounds = [cheb_reinforce,cheb_gpomdp,hoeffding,iter_bernstein,bernstein2]

    #Args: N_min, N_max, delta, estimator,bound ,outfile, MaxN
    N_min = int(sys.argv[1])
    assert N_min > 1
    N_max = int(sys.argv[2])
    assert N_max < 1000000
    delta = float(sys.argv[3])
    assert delta<1
    k = 1
    if len(sys.argv)>4:
        k = int(sys.argv[4])
    assert k<len(estimators)
    grad_estimator = estimators[k]
    k = 1
    if len(sys.argv)>5:
        k = int(sys.argv[5])
    assert k<len(bounds)
    stat_bound = bounds[k]
    print 'Using', grad_estimator.__name__, ',', stat_bound.__name__
    record = len(sys.argv) > 6
    if record:
        fp = open(sys.argv[6],'w')    
    N_maxtot = np.inf
    if len(sys.argv) > 7:
        N_maxtot = int(sys.argv[7])
 
    c = (R*M_phi**2*(gamma*math.sqrt(2*math.pi)*sigma + 2*(1-gamma)*action_volume))/ \
            (2*(1-gamma)**3*sigma**3*math.sqrt(2*math.pi))  
    
    verbose = 1    
    seed = None
    np.random.seed(seed)  

 
    #trajectory to run in parallel
    def trajectory(n,initial,noises,traces):
        s = env.reset(initial)

        for l in range(H): 
            a = np.clip(gauss_policy(s,theta,sigma,noises[l]),-env.max_action, env.max_action)
            traces[n,l,0] = gauss_score(s,a,theta,sigma)
            s,r,_,_ = env.step(a)
            traces[n,l,1] = gamma**l*r 
        

    #Learning
    J_est = J = -np.inf
    if verbose>0:
        print 'theta*:', theta_star, '\n' 
    if record:
        fp.write("{} {} {} {} {} {}\n\n".format(N_min, N_max, delta, grad_estimator.__name__,stat_bound.__name__,N_maxtot))
    iteration = 0
    path = tempfile.mkdtemp()
    traces_path = os.path.join(path,'traces.mmap')
    n_cores = multiprocessing.cpu_count() 
    assert N_min > 1
    N = N_min
    N_tot = N
    rng_emp = 0
    bad_updates = 0
    alpha = 0
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
        rng_emp = max(rng_emp,max(grad_samples) - min(grad_samples))
        rng = grad_range(R,M_phi,sigma,gamma,a_max,action_volume,theta)
        varbound = (R**2*M_phi**2)/(delta*sigma**2*(1-gamma)**2) * \
                       ((1-gamma**(2*H))/(1-gamma**2)+ H*gamma**(2*H)  - \
                            2 * gamma**H  * (1-gamma**H)/(1-gamma)) 
        d,f,f0 = stat_bound(R,M_phi,H,delta,sigma,gamma,rng,sample_var,grad_J)
           

        #Record
        if record:
            fp.write("{} {} {} {} {} {}\n".format(iteration,N,theta,alpha,J,J_est))         

        #Update
        theta+=alpha*grad_J
        
        #Adaptive batch-size (used for next batch)
        epsilon =  eps_opt(d,f,grad_J) 
        if epsilon > abs(grad_J):
            epsilon = abs(grad_J)
        print 'epsilon:', epsilon, 'grad:', grad_J, 'f:', f
        N_star = ((d + math.sqrt(d**2 + 4*epsilon*f0))/(2*epsilon))**2
        N = min(N_max,max(N_min,int(N_star) + 1))  
        
        #Adaptive step-size
        actual_eps = d/math.sqrt(N) + f0/N
        alpha = (abs(grad_J)-actual_eps)**2/(2*c*(abs(grad_J)+actual_eps)**2) 
        if verbose>0:
                print 'alpha:', alpha

        #Meta
        if verbose>0:
            print 'time:', time.time() - start, '\n'
        N_tot+=N
        if N_tot>N_maxtot:
            print "Max N reached"
            break
          
    #Cleanup 
    print '\nStopped after',iteration,'iterations, theta =',theta
    if record:
        fp.close()

