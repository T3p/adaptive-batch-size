import numpy as np
import math

#OpenAI
import gym
from trading_env import TradingEnv

#parallelism
import sys
from joblib import Parallel, delayed
import multiprocessing
import tempfile, os

#meta
import time


#Gaussian policy
def gauss_policy(s,theta,sigma,noise):
    return np.dot(s,theta) + noise*sigma

#Score for gaussian policy
def gauss_score(s,a,theta,sigma):
    return (a-np.dot(s,theta))*s/(sigma**2)

#REINFORCE gradient estimator (w/o final averaging)
def reinforce(scores,disc_rewards):
    q = np.sum(disc_rewards,1)
    sum_of_scores = np.sum(scores,1)
    #optimal baseline:
    b = np.mean(sum_of_scores**2*q)/np.mean(sum_of_scores**2)
    #gradient estimates:
    return sum_of_scores*(q-b)

#GPOMDP gradient estimator (w/out final averaging)
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

#Gradient Range: valid only in case of non-negative or non-positive reward
def grad_range(R,M_phi,sigma,gamma,a_max,action_volume):
    Q_sup = action_volume*R/(1-gamma)
    return 2*M_phi*a_max/sigma**2*Q_sup

#Generic closed form optimization for N and corresponding estimation error
def closed_opt(d,infgrad):
    eps_star = 0.25*(math.sqrt(17) - 3)*infgrad
    N_star = int(math.ceil(d**2/eps_star**2))
    return eps_star, N_star

#Optimization with Chebyshev bound for REINFORCE
def cheb_reinforce(R,M_phi,sigma,infgrad,grad_range=None,sample_var=None,c=None):
    d =  math.sqrt((R**2*M_phi**2*H*(1-gamma**H)**2)/ \
                (sigma**2*(1-gamma)**2*delta))
    
    return (d,0) + closed_opt(d,infgrad)

#Optimization with Chebyshev bound for GPOMDP
def cheb_gpomdp(R,M_phi,sigma,infgrad,grad_range=None,sample_var=None,c=None):
    d = math.sqrt((R**2*M_phi**2)/(delta*sigma**2*(1-gamma)**2) * \
                       ((1-gamma**(2*H))/(1-gamma**2)+ H*gamma**(2*H)  - \
                            2 * gamma**H  * (1-gamma**H)/(1-gamma)))
    return (d,0) + closed_opt(d,infgrad)

#Optimization with Hoeffding bound
def hoeffding(R,M_phi,sigma,infgrad,grad_range,sample_var=None,c=None):
    assert delta<1
    d = grad_range*math.sqrt(math.log(2/delta)/2)
    return (d,0) + closed_opt(d,infgrad)

#Optimizaiton with empirical Bernstein bound
def bernstein(R,M_phi,sigma,infgrad,grad_range,sample_var,c):
    assert delta<1
    d = math.sqrt(2*math.log(3.0/delta)*sample_var)
    f = 3*grad_range*math.log(3.0/delta)
    N_0 = min(N_max,max(N_min,int(((d + math.sqrt(d**2 + 4*f*infgrad)) \
            /(2*infgrad))**2) + 1))
    print 'N_0:', N_0
    ups_max = -np.inf
    eps_star = np.inf
    N_star = N_0
    for N in range(N_0,N_max+1):
        eps = d/math.sqrt(N) + f/N
        upsilon = (infgrad - eps)**4/ \
                    (4*c*(infgrad + eps)**2*N)
        if upsilon>ups_max:
            ups_max = upsilon
            eps_star = eps
            N_star = N
    return d,f,eps_star,N_star

def features(s):
    return s


if __name__ == '__main__':
    env = gym.make('trading-v0')

    #Task constants
    a_max = np.asscalar(env.action_space.high)
    a_min = np.asscalar(env.action_space.low)
    action_volume = a_max-a_min  #|A|
    gamma = 0.99
    sigma = 1 
    H = env.days
    m = 10
    #Initial policy parameter
    theta = np.zeros(m)
    
    #Options (args: N_min, N_max, delta, estimator,bound ,outfile, MaxN)
    verbose = 1 
    estimators = [reinforce,gpomdp]
    bounds = [cheb_reinforce,cheb_gpomdp,hoeffding,bernstein]
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
    k = 4
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
 
    #Trajectory (to run in parallel)
    def trajectory(n,noises,traces):
        closes = []
        obs = env.reset()
        for _ in range(m):
            closes.append(obs[0])
        for l in range(H): 
            s = np.array(closes[-1:-m-1:-1])
            traces[n,l,m+1] = max(abs(s))
            a = np.array([np.clip(gauss_policy(s,theta,sigma,noises[l]),-a_max, a_max)])
            score = gauss_score(s,a,theta,sigma)
            traces[n,l,0:m] = score 
            obs,r,_,_ = env.step(a)
            closes.append(obs[0])
            traces[n,l,m] = r  


    #LEARNING

    if record:
        fp.write("{} {} {} {} {} {}\n\n".format(N_min, N_max, delta, grad_estimator.__name__,stat_bound.__name__,N_maxtot))

    path = tempfile.mkdtemp()
    traces_path = os.path.join(path,'traces.mmap')
    grads_path = os.path.join(path,'grads.mmap')
    n_cores = multiprocessing.cpu_count() 
    
    N = N_min
    N_tot = N
    J_est = -np.inf
    bad_updates = 0
    R = M_phi = 0
    iteration = 0

    while True: 
        iteration+=1 
        if verbose > 0:
            start = time.time()
            print 'iteration:', iteration, 'N:', N, 'theta:', theta  
            
        #Run N trajectories in parallel  
        noises = np.random.normal(0,1,(N,H))
        traces = np.memmap(traces_path,dtype=float,shape=(N,H,m+2),mode='w+')  
        Parallel(n_jobs=n_cores)(delayed(trajectory)(n,noises[n],traces) for n in xrange(N))
        scores = traces[:,:,0:m]
        rewards = traces[:,:,m]
        disc_rewards = rewards
        for n in range(N):
            for l in range(H):
                disc_rewards[n,l]*=gamma**l
        feats = traces[:,:,m+1]

        #Performance estimation
        J_est0 = J_est
        J_est = np.mean(np.sum(disc_rewards,1))
        deltaJ_est = J_est - J_est0
        if iteration>1:
            if deltaJ_est<0:
                bad_updates+=1
            eff = 1-float(bad_updates)/(iteration-1)
            print 'EFF:', eff, '%' 
        if verbose>0:   
            print 'J~:', J_est
            print 'deltaJ~:', deltaJ_est
        del traces

        R = max(R,np.max(abs(rewards)))
        M_phi = max(M_phi,np.max(abs(feats)))  
        if verbose>0:
            print 'R:', R, 'M_phi:', M_phi
        rng_emp = grad_range(R,M_phi,sigma,gamma,a_max,action_volume)
        c = (R*M_phi**2*(gamma*math.sqrt(2*math.pi)*sigma + 2*(1-gamma)*action_volume))/ \
            (2*(1-gamma)**3*sigma**3*math.sqrt(2*math.pi))  
    
        #Gradient estimation
        grads_J = np.zeros(m)
        sample_vars = np.zeros(m)
    
        def compute_grads(j,grad_samples):
            grad_samples[j,:] = grad_estimator(scores[:,:,j],disc_rewards) 
    
        grad_samples = np.memmap(grads_path,dtype=float,shape=(m,N),mode='w+')  
        Parallel(n_jobs=n_cores)(delayed(compute_grads)(j,grad_samples) for j in xrange(m))
        for j in range(m):
            grads_J[j] = np.mean(grad_samples[j,:])
            sample_vars[j] = np.var(grad_samples[j],ddof=1)
        infgrad = max(abs(grads_J))
        k = np.argmax(abs(grads_J))
        sample_var = max(sample_vars)
        #rng_emp = ...
        rng = grad_range(R,M_phi,sigma,gamma,a_max,action_volume)
        d,f,eps_star,N_star = stat_bound(R,M_phi,sigma,infgrad,rng,sample_var,c)
        del grad_samples
           
        #Adaptive step-size
        actual_eps = d/math.sqrt(N) + f/N
        alpha = (infgrad - actual_eps)**2/(2*c*(infgrad + actual_eps)**2) 
        if verbose>0:
                print 'alpha:', alpha
        
        #Record
        if record:
            fp.write("{} {} {} {} {} {}\n".format(iteration,N,theta,alpha,J,J_est))         

        #Update
        theta[k]+=alpha*infgrad
        
        #Adaptive batch-size (used for next batch)
        if verbose>0:
            print 'epsilon:', eps_star, 'grad:', grads_J, 'f:', f
            if eps_star>=infgrad:
                print 'Optimal eps is too high!'
        N = min(N_max,max(N_min,N_star)) 
    
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

