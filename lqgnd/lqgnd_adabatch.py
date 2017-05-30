import numpy as np
import math

#OpenAI
import gym
from lqgnd import LQGND

#parallelism
import sys
from joblib import Parallel, delayed
import multiprocessing
import tempfile, os

#meta
import time


#Gaussian policy
def gauss_policy(s,theta,sigma,noise):
    return np.dot(theta,s) + np.dot(sigma,noise)

#Score for gaussian policy
def gauss_score(s,a,theta,sigma):
    return np.kron((a-np.dot(theta,s)),s)/(sigma**2)

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
    return 2*H*M_phi*a_max*R/(sigma**2*(1-gamma))

#Generic closed form optimization for N and corresponding estimation error
def closed_opt(d,infgrad):
    eps_star = 0.25*(math.sqrt(17) - 3)*infgrad
    N_star = int(math.ceil(d**2/eps_star**2))
    return eps_star, N_star

#Optimization with Chebyshev bound for REINFORCE
def cheb_reinforce(R,M_phi,sigma,infgrad,sample_var=None,c=None,sample_rng=None):
    d =  math.sqrt((R**2*M_phi**2*H*(1-gamma**H)**2)/ \
                (sigma**2*(1-gamma)**2*delta))
    
    return (d,0) + closed_opt(d,infgrad)

#Optimization with Chebyshev bound for GPOMDP
def cheb_gpomdp(R,M_phi,sigma,infgrad,sample_var=None,c=None,sample_rng=None):
    d = math.sqrt((R**2*M_phi**2)/(delta*sigma**2*(1-gamma)**2) * \
                       ((1-gamma**(2*H))/(1-gamma**2)+ H*gamma**(2*H)  - \
                            2 * gamma**H  * (1-gamma**H)/(1-gamma)))
    return (d,0) + closed_opt(d,infgrad)

#Optimization with Hoeffding bound
def hoeffding(R,M_phi,sigma,infgrad,sample_var=None,c=None,sample_rng=None):
    rng = grad_range(R,M_phi,sigma,gamma,a_max,action_volume)
    d = rng*math.sqrt(math.log(2/delta)/2)
    return (d,0) + closed_opt(d,infgrad)

def sample_hoeffding(R,M_phi,sigma,infgrad,sample_var,c,sample_rng):
    rng = sample_rng
    d = rng*math.sqrt(math.log(2/delta)/2)
    return (d,0) + closed_opt(d,infgrad)

def evaluate_N(N,d,f,c,infgrad):
    eps = d/math.sqrt(N) + f/N
    upsilon = (infgrad - eps)**4/ \
                (4*c*(infgrad + eps)**2*N)
    return upsilon,eps


#Optimization with empirical Bernstein bound
def bernstein(R,M_phi,sigma,infgrad,sample_var,c,sample_rng=None):
    rng = grad_range(R,M_phi,sigma,gamma,a_max,action_volume)
    d = math.sqrt(2*math.log(3.0/delta)*sample_var)
    f = 3*rng*math.log(3.0/delta)
    N_0 = min(N_max,max(N_min,int(((d + math.sqrt(d**2 + 4*f*infgrad)) \
            /(2*infgrad))**2) + 1))
    print 'N_0:', N_0
    ups_max = -np.inf
    eps_star = np.inf
    N_star = N_0
    n_cores = multiprocessing.cpu_count()
    ups,epss = zip(*Parallel(n_jobs=n_cores)(delayed(evaluate_N)(N,d,f,c,infgrad) for N in xrange(N_0,N_max+1)))
    N_star = N_0 + np.argmax(ups)
    eps_star = epss[N_star - N_0]
    return d,f,eps_star,N_star

def sample_bernstein(R,M_phi,sigma,infgrad,sample_var,c,sample_rng):
    rng = sample_rng
    d = math.sqrt(2*math.log(3.0/delta)*sample_var)
    f = 3*rng*math.log(3.0/delta)
    N_0 = min(N_max,max(N_min,int(((d + math.sqrt(d**2 + 4*f*infgrad)) \
            /(2*infgrad))**2) + 1))
    print 'N_0:', N_0
    ups_max = -np.inf
    eps_star = np.inf
    N_star = N_0
    n_cores = multiprocessing.cpu_count()
    ups,epss = zip(*Parallel(n_jobs=n_cores)(delayed(evaluate_N)(N,d,f,c,infgrad) for N in xrange(N_0,N_max+1)))
    N_star = N_0 + np.argmax(ups)
    eps_star = epss[N_star - N_0]
    return d,f,eps_star,N_star
    

if __name__ == '__main__':
    env = gym.make('LQGND-v0')

    #Task constants
    dim = env.dim
    a_max = env.max_action
    action_volume = (2*a_max)**2  #|A|
    R = np.asscalar(np.dot([env.max_pos]*dim,np.dot(env.Q,[env.max_pos]*dim))
        + np.dot([env.max_action]*dim,np.dot(env.R,[env.max_action]*dim)))
    M_phi = env.max_pos
    gamma = env.gamma 
    sigma = 1 
    H = env.horizon
    c = (R*M_phi**2*(gamma*math.sqrt(2*math.pi)*sigma + 2*(1-gamma)*action_volume))/ \
            (2*(1-gamma)**3*sigma**3*math.sqrt(2*math.pi))  
    
    #Initial policy parameter
    m = dim**2
    theta = np.zeros((dim,dim)) 

    #Optimal policy
    theta_star = env.computeOptimalK() 
    J_star = env.computeJ(theta_star,sigma,1000)
 
    #Options (args: N_min, N_max, delta, estimator,bound ,outfile, MaxN)
    verbose = 1 
    estimators = [reinforce,gpomdp]
    bounds = [cheb_reinforce,cheb_gpomdp,hoeffding,bernstein,sample_hoeffding,sample_bernstein]
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
    def trajectory(n,initial,noises,traces):
        s = env.reset(initial)
        for l in range(H): 
            a = np.clip(gauss_policy(s,theta,sigma,noises[l]),-env.max_action, env.max_action)
            traces[n,l,0:m] = np.ravel(gauss_score(s,a,theta.reshape((dim,dim)),sigma))
            s,r,_,_ = env.step(a)
            traces[n,l,m] = gamma**l*r  


    #LEARNING

    if verbose>0:
        print 'theta*:', theta_star, 'J*:', J_star, '\n' 
    if record:
        fp.write("{} {} {} {} {} {}\n\n".format(N_min, N_max, delta, grad_estimator.__name__,stat_bound.__name__,N_maxtot))

    path = tempfile.mkdtemp()
    traces_path = os.path.join(path,'traces.mmap')
    n_cores = multiprocessing.cpu_count() 
    
    N = N_min
    N_tot = N
    J_est = J = -np.inf
    rng_emp = grad_range(R,M_phi,sigma,gamma,a_max,action_volume)
    bad_updates = 0
    iteration = 0

    while True: 
        iteration+=1 
        if verbose > 0:
            start = time.time()
            print 'iteration:', iteration, 'N:', N, 'theta:', theta  
            
        #Run N trajectories in parallel  
        initials = np.random.uniform(-env.max_pos,env.max_pos,(N,dim))
        noises = np.random.normal(0,1,(N,H))
        traces = np.memmap(traces_path,dtype=float,shape=(N,H,m+1),mode='w+')  
        Parallel(n_jobs=n_cores)(delayed(trajectory)(n,initials[n],noises[n],traces) for n in xrange(N))
        scores = traces[:,:,0:m]
        disc_rewards = traces[:,:,m]

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
        theta = np.ravel(theta)
        grad_Js = []
        sample_vars = []
        sample_rngs = []
        for j in range(m):
            grad_samples = grad_estimator(scores[:,:,j],disc_rewards)
            grad_Js.append(np.mean(grad_samples))
            sample_vars.append(np.var(grad_samples,ddof=1))
            sample_rngs.append(max(grad_samples) - min(grad_samples))

        k = np.argmax(abs(np.array(grad_Js)))
        infgrad = abs(grad_Js[k])
        sample_var = max(sample_vars)
        sample_rng = max(sample_rngs)
        print 'k:', k
        
        d,f,eps_star,N_star = stat_bound(R,M_phi,sigma,infgrad,sample_var,c,sample_rng)
           
        #Adaptive step-size
        actual_eps = d/math.sqrt(N) + f/N
        alpha = (infgrad - actual_eps)**2/(2*c*(infgrad + actual_eps)**2) 
        if verbose>0:
                print 'alpha:', alpha
        
        #Record
        if record:
            fp.write("{} {} {} {} {} {}\n".format(iteration,N,theta,alpha,J,J_est))         

        #Update
        theta[k]+=grad_Js[k]*alpha 
        theta = np.reshape(theta,(dim,dim))
        
        #Adaptive batch-size (used for next batch)
        if verbose>0:
            print 'epsilon:', eps_star, 'grad:', grad_Js, 'f:', f
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

