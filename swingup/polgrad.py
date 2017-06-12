import numpy as np
import math
from numbers import Number

#OpenAI
import gym

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


if __name__ == '__main__': 
    #Options (args: environment,theta0,N,alpha,estimator,outfile,MaxN)
    verbose = 1 
    perform = True
    estimators = [reinforce,gpomdp]
    #Task
    task = 'Pendulum-v0'
    if len(sys.argv)>1:
        task = sys.argv[1]
    env = gym.make(task)
    #Initial param
    s_dim = len(env.reset())
    a_dim = env.action_space.shape[0]
    theta = np.ravel(np.zeros((s_dim,a_dim)))
    if len(sys.argv)>2:
        raw = np.fromstring(sys.argv[2],sep=' ')
        theta = np.ravel(raw.reshape(np.shape(theta)))
    #Batch size
    N = 500
    if len(sys.argv)>3:
        N = int(sys.argv[3])
    assert N > 1
    #Step size
    alpha = 1e-5
    if len(sys.argv)>4:
        alpha = float(sys.argv[4])
    #Gradient estimator
    k = 1
    if len(sys.argv)>5:
        k = int(sys.argv[5])
    assert k<len(estimators)
    grad_estimator = estimators[k]
    #Output file
    record = True
    filename = 'log.out'
    if len(sys.argv) > 6:
        filename = sys.argv[6]
    fp = open(filename,'w')
    #Experiment length    
    N_maxtot = np.inf
    if len(sys.argv) > 7:
        N_maxtot = int(sys.argv[7])   
    

    #Task constants
    a_max = env.action_space.high
    if not isinstance(a_max,Number):
        a_max = max(a_max)
    action_volume = env.action_space.high-env.action_space.low
    if not isinstance(action_volume,Number):
        action_volume = reduce(lambda x,y: x*y,action_volume)
    R = M_phi = sample_rng = 0 
    gamma = 0.99 
    sigma = np.eye(a_dim)*1.0 
    H = 200
    m = len(theta)


    #Trajectory (to run in parallel)
    def trajectory(n,seed,noises,traces):
        env.seed(seed)
        s = env.reset()
        max_s = abs(s) if isinstance(s,Number) else max(abs(s))

        for l in range(H): 
            s = s.reshape((s_dim,1))
            a = np.clip(gauss_policy(s,theta.reshape(s_dim,a_dim),sigma,noises[l]),env.action_space.low, env.action_space.high)
            traces[n,l,0:m] = np.ravel(gauss_score(s,a,theta.reshape((s_dim,a_dim)),sigma))
            s,r,_,_ = env.step(a)
            traces[n,l,m] = r  
    
            max_s = max(max_s,abs(s)) if isinstance(s,Number) else max(max_s,max(abs(s)))
    
        return max_s


    def show():
        s = env.reset()
        for l in range(H): 
            s = s.reshape((s_dim,1))
            a = np.clip(gauss_policy(s,theta.reshape(s_dim,a_dim),sigma,np.random.normal(0,1,a_dim)),env.action_space.low, env.action_space.high)
            s,_,_,_ = env.step(a)
            env.render()
        

    #LEARNING
    if verbose>0:
        print 'Using', grad_estimator.__name__
    if record:
        fp.write("{} {} {} {}\n\n".format(N,alpha,grad_estimator.__name__,N_maxtot))
    
    #Parallelization setup
    path = tempfile.mkdtemp()
    traces_path = os.path.join(path,'traces.mmap')
    n_cores = multiprocessing.cpu_count() 


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
               
        #Record
        if record:
            fp.write("{} {} {}\n".format(iteration,theta,J))         

        #Update
        theta+=np.array(grad_Js)*alpha
        
        #Log
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

