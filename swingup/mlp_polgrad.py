import numpy as np
import tensorflow as tf
import math
from numbers import Number
from mlppolicy import NormalPolicy
#from swingPendulum import SwingPendulum
from pendulum_edit import PendulumEnv

#OpenAI
import gym

#parallelism
import sys
from joblib import Parallel, delayed
import multiprocessing
import tempfile, os

#meta
import time


#REINFORCE gradient estimator (w/o final averaging)
def reinforce(scores,disc_rewards):
    q = np.sum(disc_rewards,1)
    sum_of_scores = np.sum(scores,1)
    #optimal baseline:
    b = np.mean(sum_of_scores**2*q)/np.mean(sum_of_scores**2)
    #gradient estimates:
    return np.mean(sum_of_scores*(q-b))

#GPOMDP gradient estimator
def gpomdp(scores,disc_rewards):
    N = scores.shape[0]
    H = scores.shape[1]
    cumulative_scores = np.zeros((N,H))
    #optimal baseline:
    b = np.zeros(H)
    for k in range(0,H):
        cumulative_scores[:,k] = sum(scores[:,i] for i in range(0,k+1))
        den = np.mean(cumulative_scores[:,k]**2)
        if den!=0:
            b[k] = np.mean(cumulative_scores[:,k]**2*disc_rewards[:,k])/ \
                        den
    #gradient estimate:
    return np.mean(sum(cumulative_scores[:,i]*(disc_rewards[:,i] - b[i]) for i in range(0,H)))


if __name__ == '__main__': 
    #Options (args: environment,theta0,N,alpha,estimator,outfile,MaxN)
    verbose = 1
    render = False 
    perform = True
    estimators = [reinforce,gpomdp]
    #Task
    task = 'Pendulum_edit-v0'
    if len(sys.argv)>1:
        task = sys.argv[1]
    env = gym.make(task)
    #Initial param
    #TODO
    #Batch size
    N = 100
    if len(sys.argv)>3:
        N = int(sys.argv[3])
    #assert N > 1
    #Step size
    alpha = 0.001
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
    N_maxtot = 100000#np.inf
    if len(sys.argv) > 7:
        N_maxtot = int(sys.argv[7])   
    
    #Agent's state
    def feat(s):
        return s

    #Task constants
    s_dim = len(feat(env.reset()))
    a_dim = env.action_space.shape[0]
    a_max = env.action_space.high
    if not isinstance(a_max,Number):
        a_max = max(a_max)
    print a_max
    action_volume = env.action_space.high-env.action_space.low
    if not isinstance(action_volume,Number):
        action_volume = reduce(lambda x,y: x*y,action_volume)
    gamma = 0.99
    sigma = 0.5 
    alpha = 1e-4
    H = 200
    N = 100


    #Tweak task
    #env.cost*=2
    env.m*=1.1

    #Policy (Multi-layer perceptron)
    mu_hidden = [400,300,50]
    mu_activ = [tf.nn.relu,tf.nn.relu,tf.nn.tanh]
    state_var = tf.placeholder(tf.float32, [1,s_dim])
    action_var = tf.placeholder(tf.float32, [1,a_dim])
    pol = NormalPolicy(1,mu_hidden,mu_activ,[],env.action_space.low,env.action_space.high,
                min_std = sigma,fixed_std=True)(state_var,action_var)
    M_phi = 1

    #Trajectory (to run in parallel)
    def trajectory(n,traces):
        s = feat(env.reset())
        #env.state = np.array([np.pi,0]) + np.array([np.random.normal(0,0.1),0])
        #s = env._get_obs()
        ret = 0

        for l in range(H): 
            s_feed = s.reshape((1,s_dim))
            mu = pol.get_mu(s_feed) 
            a = np.clip(mu + np.dot(sigma,np.random.normal(0,1,a_dim)),env.action_space.low, env.action_space.high)
            a_feed = a.reshape((1,a_dim))
            score = pol.outer_log_gradients(s_feed,a_feed)
            traces[n,l,0:m] = score
            obs,r,_,_ = env.step(a)
            s = feat(obs)
            traces[n,l,m] = r
            ret+=gamma**l*r
            if render:
                env.render()
      
        return ret


    def show():
        np.random.seed(42)
        env.seed(42)
        s = feat(env.reset())
        env.state = np.array([np.pi,0])
        s = env._get_obs()
        ret = 0
        for l in range(H):
            s_feed = s.reshape((1,s_dim)) 
            mu = pol.get_mu(s_feed)
            a = np.clip(mu + np.dot(sigma,np.random.normal(0,1,a_dim)),env.action_space.low, env.action_space.high)
            obs,r,_,_ = env.step(a)
            s = feat(obs)
            #env.render()
            ret += gamma**l*r
        return ret

    #LEARNING
    if verbose>0:
        print 'Using', grad_estimator.__name__
    if record:
        fp.write("{} {} {} {}\n\n".format(N,alpha,grad_estimator.__name__,N_maxtot))
    
    #Parallelization setup
    path = tempfile.mkdtemp()
    traces_path = os.path.join(path,'traces.mmap')
    n_cores = multiprocessing.cpu_count() 

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        theta = pol.get_outer_weights()
        m = len(theta)
        pol.load_weights('weights/dpg_f50_sigma0.5_it100.npy')  
   
        R = sample_rng = 0 
        N_tot = N
        J = -np.inf
        J_start = 0
        it_0 = 0
        iteration = it_0
        while True: 
            iteration+=1 
            pol.save_weights('weights/polgrad')
            if verbose > 0:
                start = time.time()
                theta = pol.get_outer_weights()
                print 'iteration:', iteration, 'N:', N, 'theta:', max(theta)  
                
            #Show current performance
            J_test = show() if perform else 0

            #Run N trajectories in parallel  
            print 'COLLECTING SAMPLES'
            traces = np.memmap(traces_path,dtype=float,shape=(N,H,m+1),mode='w+')  
            rets = [trajectory(n,traces) for n in xrange(N)]
            scores = traces[:,:,0:m]
            rewards = traces[:,:,m]
            R = max(R,np.max(abs(rewards)))
            disc_rewards = np.zeros((N,H))
            for n in range(N):
                for l in range(H):
                    disc_rewards[n,l] = rewards[n,l]*sigma**l

            #Performance estimation
            J0 = J
            J = np.mean(rets)
            
            if iteration>it_0+1:
                deltaJ = J - J0
                tot_deltaJ = J - J_start
            else:
                deltaJ = 0
                tot_deltaJ = 0
                J_start = J

            if verbose>0:   
                #print 'Jtest:', J_test
                print 'J~:', J
                print 'deltaJ~:', deltaJ
                print 'tot_deltaJ~:', tot_deltaJ
            del traces
            
            #Gradient estimation 
            print 'COMPUTING GRADIENTS' 
            grad_Js = Parallel(n_jobs=n_cores)(delayed(grad_estimator)(scores[:,:,j],disc_rewards) 
                        for j in xrange(m))
            print 'Grads:', max(grad_Js)
                   
            #Record
            if record:
                fp.write("{} {} {} {}\n".format(iteration,theta,J,J_test))         

            #Update
            k = np.argmax(abs(np.array(grad_Js)))
            alpha_t = np.zeros(m)
            alpha_t[k] = alpha#float(alpha)/math.sqrt(iteration)
            pol.update_outer_layer(alpha*np.array(grad_Js))
            
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

