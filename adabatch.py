#parallelism
import sys
from joblib import Parallel,delayed
import multiprocessing
import tempfile,os

#adabatch
from policies import GaussPolicy
from meta_optimization import *
from gradient_estimation import performance, Estimator

import time
import signal
import tables
from utils import zero_fun


#Trajectory (can be run in parallel)
def __trajectory(env,tp,pol,feature_fun,traces,n,initial=None,noises=[]):
    if  len(noises)==0:
        noises = np.random.normal(0,1,tp.H)

    s = env.reset(initial)
    for l in range(tp.H): 
        phi = feature_fun(s)
        a = np.clip(pol.act(phi,noises[l]),tp.min_action,tp.max_action)
        s,r,_,_ = env.step(a)
        traces[n,l] = np.concatenate((np.atleast_1d(phi),np.atleast_1d(a),np.atleast_1d(r)))

def learn(env,tp,pol,feature_fun,constr,bound_name='bernstein',estimator_name='gpomdp',emp=True,evaluate=zero_fun,parallel=True,filepath='results/record.h5',verbose=1):
    """
        Vanilla policy gradient with adaptive step size and batch size
        
        Parameters:
        env -- the gym environment
        tp -- TaskProp object with task properties
        pol -- parametric policy
        feature_fun -- feature function
        constr -- constraints on the meta-optimization
        bound -- statistical bound used to compute the batch size
        estimator -- gradient estimation algorithm
        empirical -- use empirical range or not
        parallel -- parallelize using joblib or not
        verbose -- how much printing
    """
    #Initial batch size
    N = N_old = constr.N_min   

    #Optimizer settings
    grad_estimator = Estimator(estimator_name)
    meta_selector = MetaOptimizer(bound_name,constr,estimator_name,emp)   

    #Multiprocessing preparation
    if parallel:
        path = tempfile.mkdtemp()
        traces_path = os.path.join(path,'traces.mmap')
        n_cores = multiprocessing.cpu_count()

    #Record
    global theta_save 
    entry_size = 5
    fp = tables.open_file(filepath,mode='w')
    atom = tables.Float32Atom()
    record = fp.create_earray(fp.root,'data',atom,(0,entry_size))

    #Initial print
    if verbose:
        print 'Estimator: ', estimator_name,  ' Bound: ', bound_name,  ' Empirical range: ', emp,  ' delta =', constr.delta
        print 'Start Experiment'
        print  

    #Learning 
    iteration = 0
    N_tot = 0
    while True:
        iteration+=1

        #Print before
        if verbose:
            print 'Epoch: ', iteration,  ' N =', N,  ' theta =', pol.get_theta()
            start_time = time.time()
    
        #Collecting experience
        if parallel:
            initials = np.random.uniform(tp.min_state,tp.max_state,N)
            noises = np.random.normal(0,1,(N,tp.H))
            traces = np.memmap(traces_path,dtype=float,shape=(N,tp.H,pol.feat_dim+pol.act_dim+1),mode='w+')
            Parallel(n_jobs=n_cores)(delayed(__trajectory)\
                (env,tp,pol,feature_fun,traces,n,initials[n],noises[n]) for n in xrange(N))
        else:
            traces = np.zeros((N,tp.H,pol.feat_dim+pol.act_dim+1))
            for n in xrange(N):
                __trajectory(env,tp,pol,feature_fun,traces,n)
        features = traces[:,:,:pol.feat_dim]
        actions = traces[:,:,pol.feat_dim:pol.feat_dim+pol.act_dim]
        rewards = traces[:,:,-1]    

        #Gradient statistics
        grad_samples = grad_estimator.estimate(features,actions,rewards,tp.gamma,pol,average=False)
        g_stats = GradStats(grad_samples)

        #Performance statistics 
        J_hat = performance(rewards,tp.gamma)
        J = evaluate(pol) 

        #Meta-optimization
        alpha,N,safe = meta_selector.select(pol,g_stats,tp,N_old)
        if not safe and verbose:
            print "Unsafe update!"

        #Record [N, alpha, k, J, J^]
        k = g_stats.get_amax()
        entry = np.array([[N_old,alpha[k],k,J,J_hat]])
        record.append(entry)
        N_old = N

        #Check if done
        N_tot+=N
        if N_tot>=constr.N_tot:
            print 'Total N reached'
            print 'End experiment'
            break

        #Optimization
        pol.update(alpha*g_stats.get_estimate()) 
        theta_save = pol.get_theta()

        #Print after
        if verbose:
            print 'alpha =', alpha,  ' J =', J,  ' J^ =', J_hat
            print 'time: ', time.time() - start_time
            print

        #Manual stop
        signal.signal(signal.SIGINT, signal_handler)
    
    #Cleanup
    np.save('theta.npy',theta_save)
    fp.close()


#Handle Ctrl-C
def signal_handler(signal,frame):
    np.save('theta.npy',theta_save)
    sys.exit(0)

