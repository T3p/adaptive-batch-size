import numpy as np
import math
from numbers import Number

#OpenAI
import gym
from cartpole import ContCartPole

#parallelism
import sys
from joblib import Parallel, delayed
import multiprocessing
import tempfile, os

#meta
import time


#Gaussian policy
def gauss_policy(s,theta,sigma,noise):
    return np.dot(theta.T,s) + np.dot(sigma,noise)

#Score for gaussian policy
def gauss_score(s,a,theta,sigma):
    return np.kron((a-np.dot(theta.T,s)),s)/np.dot(sigma.T,sigma)

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
    for n in range(N_0,N_max):
        ups,eps = evaluate_N(n,d,f,c,infgrad)
        if ups>ups_max:
            ups_max = ups
            eps_star = eps
            N_star = n
        else:
            break
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
    for n in range(N_0,N_max):
        ups,eps = evaluate_N(n,d,f,c,infgrad)
        if ups>ups_max:
            ups_max = ups
            eps_star = eps
            N_star = n
        else:
            break
    return d,f,eps_star,N_star    


if __name__ == '__main__': 
    #Options (args: environment,N_min, N_max, delta, estimator,bound ,outfile, MaxN)
    verbose = 1 
    perform = False
    estimators = [reinforce,gpomdp]
    bounds = [cheb_reinforce,cheb_gpomdp,hoeffding,bernstein,sample_hoeffding,sample_bernstein]
    #Task
    task = 'ContCartPole-v0'
    if len(sys.argv)>1:
        task = sys.argv[1]
    env = gym.make(task)
    #Initial param
    s_dim = len(env.reset())
    a_dim = env.action_space.shape[0]
    theta = np.load('theta_star.npy')#np.ravel(np.zeros((s_dim,a_dim)))
    if len(sys.argv)>2:
        raw = np.fromstring(sys.argv[2],sep=' ')
        theta = np.ravel(raw.reshape(np.shape(theta)))
    #Min batch size
    N_min = 10
    if len(sys.argv)>3:
        N_min = int(sys.argv[3])
    assert N_min > 1
    #Max batch size
    N_max = 500000
    if len(sys.argv)>4:
        N_max = int(sys.argv[4])
    assert N_max < 1000000
    #Worsening probability
    delta = 0.95
    if len(sys.argv)>5:
        delta = float(sys.argv[5])
    assert delta<1
    #Gradient estimator
    k = 1
    if len(sys.argv)>6:
        k = int(sys.argv[6])
    assert k<len(estimators)
    grad_estimator = estimators[k]
    #Statistical bound
    k = 5
    if len(sys.argv)>7:
        k = int(sys.argv[7])
    assert k<len(bounds)
    stat_bound = bounds[k]
    #Output file
    record = True
    fp = open('results/long_tweak2_adabatch.out','w')
    #record = len(sys.argv) > 8
    #if record:
        #fp = open(sys.argv[8],'w')
    #Experiment length    
    N_maxtot = 3000000
    #if len(sys.argv) > 9:
        #N_maxtot = int(sys.argv[7])   
    

    #Task constants
    a_max = env.action_space.high
    if not isinstance(a_max,Number):
        a_max = max(a_max)
    action_volume = env.action_space.high-env.action_space.low
    if not isinstance(action_volume,Number):
        action_volume = reduce(lambda x,y: x*y,action_volume)
    R = M_phi = sample_rng = 0 
    gamma = 0.9 
    sigma = np.eye(a_dim)*1.0 
    H = 200
    m = len(theta)

    #Task tweaking
    env.length*=2.0

    #Trajectory (to run in parallel)
    def trajectory(n,seed,noises,traces):
        env.seed(seed)
        s = env.reset()
        max_s = abs(s) if isinstance(s,Number) else max(abs(s))

        for l in range(H): 
            s = s.reshape((s_dim,1))
            a = np.clip(gauss_policy(s,theta.reshape(s_dim,a_dim),sigma,noises[l]),env.action_space.low, env.action_space.high)
            traces[n,l,0:m] = np.ravel(gauss_score(s,a,theta.reshape((s_dim,a_dim)),sigma))
            s,r,done,_ = env.step(a)
            traces[n,l,m] = r  
    
            max_s = max(max_s,abs(s)) if isinstance(s,Number) else max(max_s,max(abs(s)))
            if done:
                break    

        return max_s


    def show():
        s = env.reset()
        for l in range(H): 
            s = s.reshape((s_dim,1))
            a = np.clip(gauss_policy(s,theta.reshape(s_dim,a_dim),sigma,np.random.normal(0,1,a_dim)),-env.action_space.low, env.action_space.high)
            s,_,done,_ = env.step(a)
            env.render()
            if done:
                break
        

    #LEARNING
    if verbose>0:
        print 'Using', grad_estimator.__name__, ',', stat_bound.__name__
    if record:
        fp.write("{} {} {} {} {} {}\n\n".format(N_min, N_max, delta, grad_estimator.__name__,stat_bound.__name__,N_maxtot))
    
    #Parallelization setup
    path = tempfile.mkdtemp()
    traces_path = os.path.join(path,'traces.mmap')
    n_cores = multiprocessing.cpu_count() 

    N = 1000#N_min
    N_tot = N
    J = -np.inf
    iteration = 0
    while True: 
        iteration+=1 
        if verbose > 0:
            start = time.time()
            print 'iteration:', iteration, 'N:', N, 'theta:', theta  
            
        #Show current performance
        if perform:
            show()

        #Run N trajectories in parallel  
        print 'COLLECTING SAMPLES'
        seeds = np.random.randint(N,size=(N,))
        noises = np.random.normal(0,1,(N,H,a_dim))
        traces = np.memmap(traces_path,dtype=float,shape=(N,H,m+1),mode='w+')  
        max_obs = Parallel(n_jobs=n_cores)(delayed(trajectory)(n,seeds[n],noises[n],traces) for n in xrange(N))
        scores = traces[:,:,0:m]
        rewards = traces[:,:,m]
        R = np.max(abs(rewards)) #max(R,rewards)
        M_phi = max(max_obs) #max(M_phi,rewards)
        disc_rewards = np.zeros((N,H))
        for n in range(N):
            for l in range(H):
                disc_rewards[n,l] = rewards[n,l]*sigma**l

        #Performance estimation
        J0 = J
        J = np.mean(np.sum(disc_rewards,1))
        
        if iteration>1:
            deltaJ = J - J0
            tot_deltaJ = J - J_start
        else:
            deltaJ = 0
            tot_deltaJ = 0
            J_start = J

        if verbose>0:   
            print 'J~:', J
            print 'deltaJ~:', deltaJ
            print 'tot_deltaJ~:', tot_deltaJ
        del traces
        
        #Gradient estimation 
        print 'COMPUTING GRADIENTS'
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
        if infgrad==0:
            print 'zero gradient!'
            break
        sample_var = max(sample_vars)
        sample_rng = max(sample_rngs) #max(sample_rng,max(sample_rngs))
        print 'k:', k
        
        #Meta-optimization
        print 'META-OPTIMIZING'
        c = (R*M_phi**2*(gamma*math.sqrt(2*math.pi)*sigma + 2*(1-gamma)*action_volume))/ \
                (2*(1-gamma)**3*sigma**3*math.sqrt(2*math.pi))  
        d,f,eps_star,N_star = stat_bound(R,M_phi,sigma,infgrad,sample_var,c,sample_rng)
        actual_eps = d/math.sqrt(N) + f/N
        alpha = (infgrad - actual_eps)**2/(2*c*(infgrad + actual_eps)**2) 
        if verbose>0:
                print 'alpha:', alpha
        
        #Record
        if record:
            fp.write("{} {} {} {} {}\n".format(iteration,N,theta,alpha,J))         

        #Update
        theta[k]+=grad_Js[k]*alpha
        
        #Adaptive batch-size (used for next batch)
        if verbose>0:
            print 'epsilon:', eps_star, 'grad:', grad_Js, 'f:', f
            if eps_star>=infgrad:
                print 'Optimal eps is too high! No guarantees!'
        N = min(N_max,max(N_min,N_star)) 
    
        print 'Ntot:', N_tot
        #Log
        if verbose>0:
            print 'time:', time.time() - start, '\n'
        N_tot+=N
        if N_tot>N_maxtot:
            print "Max N reached"
            break
        if J>195:
            print "Solved!"
            break
          
    #Cleanup 
    print '\nStopped after',iteration,'iterations, theta =',theta
    if record:
        fp.close()

