import numpy as np
import math
import tensorflow as tf
from mlppolicy import NormalPolicy

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
    b = 0
    score_mean = np.mean(sum_of_scores**2)
    if score_mean!=0:
        b = np.mean(sum_of_scores**2*q)/score_mean
    #gradient estimates:
    return sum_of_scores*(q-b)

#GPOMDP gradient estimator (w/out final averaging)
def gpomdp(scores,disc_rewards):
    N = scores.shape[0]
    H = scores.shape[1]
    m = scores.shape[2]
    cumulative_scores = np.zeros((N,H,m))

    #optimal baseline:
    b = np.zeros((H,m))
    for k in range(0,H):
        cumulative_scores[:,k,:] = sum(scores[:,i,:] for i in range(0,k+1))
        cumul_score_mean = np.mean(cumulative_scores[:,k,:]**2,0)
        for j in range(m):
            if cumul_score_mean[j]!=0:
                b[k,j] = np.mean(cumulative_scores[:,k,j]**2*disc_rewards[:,k])/ \
                    cumul_score_mean[j] 
    #gradient estimate:
    return np.array(
            [sum(cumulative_scores[:,i,j]*(disc_rewards[:,i] - b[i,j]) for i in range(0,H)) 
                for j in range(m)])

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

#Optimization with empirical Bernstein bound
def bernstein(R,M_phi,sigma,infgrad,sample_var,c,sample_rng=None):
    rng = grad_range(R,M_phi,sigma,gamma,a_max,action_volume)
    d = math.sqrt(2*math.log(3.0/delta)*sample_var)
    f = 3*rng*math.log(3.0/delta)
    N_0 = min(N_max,max(N_min,int(((d + math.sqrt(d**2 + 4*f*infgrad)) \
            /(2*infgrad))**2) + 1))
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

def sample_bernstein(R,M_phi,sigma,infgrad,sample_var,c,sample_rng):
    rng = sample_rng
    d = math.sqrt(2*math.log(3.0/delta)*sample_var)
    f = 3*rng*math.log(3.0/delta)
    N_0 = min(N_max,max(N_min,int(((d + math.sqrt(d**2 + 4*f*infgrad)) \
            /(2*infgrad))**2) + 1))
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


if __name__ == '__main__': 
    N = 10
    alpha = 1e-2 
    neurons_per_input = 32
    hidden_layers = 1 

    env = gym.make('trading-v0')

    #Task constants
    a_max = np.asscalar(env.action_space.high)
    a_min = np.asscalar(env.action_space.low)
    action_volume = a_max-a_min  #|A|
    gamma = 0.9
    
    #Normal policy with 2-layers NN mean and fixed std
    obs_size = 5
    n_obs = 1
    action_size = 1 #scalar action
    sigma = 0.001
    hidden_neurons = obs_size*neurons_per_input
    state_var = tf.placeholder(tf.float32, [n_obs,obs_size])
    action_var = tf.placeholder(tf.float32, [n_obs,action_size])
    pol = NormalPolicy(1,[hidden_neurons]*hidden_layers,[],a_min,a_max, \
                min_std=sigma,fixed_std=True)(state_var,action_var)
    H = env.days
    print 'Trajectory size:', H
     
    #Options (args: delta, N_min, N_max, estimator,bound ,outfile, MaxN)
    verbose = 1 
    estimators = [reinforce,gpomdp]
    bounds = [cheb_reinforce,cheb_gpomdp,sample_hoeffding,sample_bernstein]
    #delta = 0.95
    #if len(sys.argv)>1:
    #    delta = float(sys.argv[1])
    #assert delta<1
    #N_min = 2
    #if len(sys.argv)>2:
    #    N_min = int(sys.argv[2])
    #assert N_min > 1
    #N_max = 500000
    #if len(sys.argv)>3:
    #    N_max = int(sys.argv[3])
    #assert N_max < 1000000
    k = 1
    #if len(sys.argv)>4:
    #    k = int(sys.argv[4])
    #assert k<len(estimators)
    grad_estimator = estimators[k]
    #k = 3
    #if len(sys.argv)>5:
    #    k = int(sys.argv[5])
    #assert k<len(bounds)
    #stat_bound = bounds[k]
    #print 'Using', grad_estimator.__name__, ',', stat_bound.__name__
    record = len(sys.argv) > 1
    if record:
        fp = open(sys.argv[1],'w')    
    N_maxtot = np.inf
    if len(sys.argv) > 2:
        N_maxtot = int(sys.argv[2])  
 

    #LEARNING

    if record:
        fp.write("{} {} {} {}\n\n".format(N,alpha,hidden_neurons,N_maxtot))

    path = tempfile.mkdtemp()
    traces_path = os.path.join(path,'traces.mmap')
    grads_path = os.path.join(path,'grads.mmap')
    n_cores = multiprocessing.cpu_count() 
    
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer()) 
        #Initial policy parameter
        theta = pol.get_weights()
        m = len(theta)
        pol.reset(np.random.randn(m)*0.05)
        
        #N = N_min
        N_tot = N
        J_est = -np.inf
        deltaJ_tot = 0
        bad_updates = 0
        R = M_phi = 0
        iteration = 0
        gain = 0
        while True: 
            iteration+=1 
            if verbose > 0:
                start = time.time()
                print 'iteration:', iteration, 'N:', N, #'theta:', theta  
                
            #Run N trajectories in parallel  
            #traces = np.memmap(traces_path,dtype=float,shape=(N,H,m+1),mode='w+')  
            #max_states = Parallel(n_jobs=n_cores)(delayed(trajectory)(n,noises[n],traces) for n in xrange(N))

            max_s = max_r = 0
            scores = np.zeros((N,H,m))
            disc_rewards = np.zeros((N,H))
            navs = []

            for n in range(N):
                s = env.reset()

                for l in range(H): 
                    s_feed = s.reshape((n_obs,obs_size))
                    mu = np.asscalar(pol.get_mu(s_feed))
                    a = np.clip(mu + sigma*np.random.randn(),a_min,a_max)
                    a_feed = a.reshape((n_obs,action_size))
                    scores[n,l] = pol.log_gradients(s_feed,a_feed)
                    s,r,_,info = env.step(np.array([a]))
                    max_s = max(max_s,max(abs(s)))
                    max_r = max(max_r,abs(r))
                    disc_rewards[n][l] = r*gamma**l
                
                navs.append(info['nav'])
            
            nav = np.mean(navs)
            gain += sum(np.array(navs) -1)
            print 'Gain:', gain
            print 'Nav at the end:', nav

            #Performance estimation
            J_est0 = J_est
            J_est = np.mean(np.sum(disc_rewards,1))
            deltaJ_est = J_est - J_est0
            if iteration>1:
                deltaJ_tot+=deltaJ_est
                if deltaJ_est<0:
                    bad_updates+=1
                eff = 1-float(bad_updates)/(iteration-1)
                print 'EFF:', eff, '%' 
                print 'tot_improv:', deltaJ_tot
            if verbose>0:   
                print 'J~:', J_est
                print 'deltaJ~:', deltaJ_est

            #R = max(R,max_r)
            #M_phi = max(M_phi,max_s)
            #if verbose>0:
            #    print 'R:', R, 'M_phi:', M_phi
            #c = (R*M_phi**2*(gamma*math.sqrt(2*math.pi)*sigma + 2*(1-gamma)*action_volume))/ \
            #    (2*(1-gamma)**3*sigma**3*math.sqrt(2*math.pi))  
        
            #Gradient estimation
            print 'Computing gradients'
            grad_samples = grad_estimator(scores,disc_rewards)
            grads_J = np.mean(grad_samples,axis=1)
            #sample_vars = np.var(grad_samples,axis=1,ddof=1) 
            #grad_abs = abs(grads_J)

            #infgrad = max(grad_abs)
            #print 'maxgrad:', infgrad
            #k = np.argmax(grad_abs)
            #sample_var = max(sample_vars)
            #rng = max([max(grad_samples[j,:]) - min(grad_samples[j,:]) for j in range(m)]) 
            #print 'Range:', rng
            #d,f,eps_star,N_star = stat_bound(R,M_phi,sigma,infgrad,sample_var,c,rng)
            #del grad_samples
               
            #Adaptive step-size
            #actual_eps = d/math.sqrt(N) + f/N
            #alpha = (infgrad - actual_eps)**2/(2*c*(infgrad + actual_eps)**2) 
            #if verbose>0:
            #        print 'alpha:', alpha
            
            #Record
            if record:
                fp.write("{} {} {} {}\n".format(iteration,J_est,nav,deltaJ_tot))         

            #Update
            #alpha_vect = np.zeros((m,),dtype=np.float32)
            #alpha_vect[k] = alpha
            alpha_vect = alpha*np.ones((m,),dtype=np.float32)
            pol.update(grads_J*alpha_vect)
            
            #Adaptive batch-size (used for next batch)
            #if verbose>0:
            #    print 'epsilon:', eps_star, 'maxgrad:', infgrad, 'f:', f
            #    if eps_star>=infgrad:
            #        print 'eps too high!'
            #N = min(N_max,max(N_min,N_star)) 
            #print 'Next N:', N
        
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

