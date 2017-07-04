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
    alpha = 1e-7
    H = 200
    N = 12


    #Tweak task
    #env.cost*=1.
    env.m*=1.05

    #Policy (Multi-layer perceptron)
    mu_hidden = [400,300,7]
    mu_activ = [tf.nn.relu,tf.nn.relu,tf.nn.tanh]
    state_var = tf.placeholder(tf.float32, [1,s_dim])
    action_var = tf.placeholder(tf.float32, [1,a_dim])
    pol = NormalPolicy(1,mu_hidden,mu_activ,[],env.action_space.low,env.action_space.high,
                min_std = sigma,fixed_std=True)(state_var,action_var)
    M_phi = 1

    def show(s_0,seed):
        np.random.seed(seed)
        env.seed(seed)
        s = feat(env.reset())
        env.state = s_0 #pi,0
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
        pol.load_weights('weights/dpg_f7_sigma0_it1000.npy')  
        #pol.update_outer_layer(-np.array(pol.get_outer_weights())+np.random.normal(0,0.14,m))
   
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
                

            #Run N trajectories 
            print 'COLLECTING SAMPLES'
            traces = np.zeros((N,H,s_dim+a_dim+1))
            for n in range(N):
                s = feat(env.reset())

                for l in range(H): 
                    s_feed = s.reshape((1,s_dim))
                    traces[n,l,0:s_dim] = np.ravel(s)
                    mu = pol.get_mu(s_feed) 
                    a = np.clip(mu + np.dot(sigma,np.random.normal(0,1,a_dim)),env.action_space.low, env.action_space.high)
                    traces[n,l,s_dim:s_dim+a_dim] = a
                    obs,r,_,_ = env.step(a)
                    traces[n,l,s_dim+a_dim] = r
                    s = feat(obs)  
            
            scores = np.zeros((N,H,m))
            disc_rewards = np.zeros((N,H))
            for n in range(N):
                for l in range(H):
                    scores[n,l] = pol.outer_log_gradients(traces[n,l,0:s_dim].reshape((1,s_dim)),traces[n,l,s_dim:s_dim+a_dim].reshape((1,a_dim)))
                    disc_rewards[n,l] = gamma**l*traces[n,l,s_dim+a_dim]

           #Performance estimation
            print 'TESTING'
            J0 = J
            initials = np.linspace(0,2*np.pi,12)
            seed = 42
            rets = [show(np.array([ang,0]),seed) for ang in initials]
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
                fp.write("{} {} {}\n".format(iteration,theta,J))         

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

