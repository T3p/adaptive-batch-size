import gym
from lqg1d import LQG1D
import numpy as np
import math
import sys

def gauss_policy(s,theta,sigma,noise):
    return s*theta + noise*sigma

def gauss_score(s,a,theta,sigma):
    return (a-s*theta)*s/(sigma**2)

def reinforce_grad(scores,disc_rewards):
    q = np.sum(disc_rewards,1)
    sum_of_scores = np.sum(scores,1)
    #optimal baseline:
    b = np.mean(sum_of_scores**2*q)/np.mean(sum_of_scores**2)
    #gradient estimate:
    return np.mean(sum_of_scores*(q-b))

def gpomdp_grad(scores,disc_rewards):
    H = scores.shape[1]
    cumulative_scores = np.zeros((N,H))
    #optimal baseline:
    b = np.zeros(H)
    for k in range(0,H):
        cumulative_scores[:,k] = sum(scores[:,i] for i in range(0,k+1))
        b[k] = np.mean(cumulative_scores[:,k]**2*disc_rewards[:,k])/ \
                    np.mean(cumulative_scores[:,k]**2)
    #gradient estimate:
    return np.mean(sum(cumulative_scores[:,i]*(disc_rewards[:,i] - b[i]) for i in range(0,H)))
       
def reinforce_d(R,M_phi,H,delta,sigma,gamma):
    return math.sqrt((R**2*M_phi**2*H*(1-gamma**H)**2)/ \
                (sigma**2*(1-gamma)**2*delta)) 

def gpomdp_d(R,M_phi,H,delta,sigma,gamma):
    return math.sqrt((R**2*M_phi**2)/(delta*sigma**2*(1-gamma)**2) * \
                       ((1-gamma**(2*H))/(1-gamma**2)+ H*gamma**(2*H)  - \
                            2 * gamma**H  * (1-gamma**H)/(1-gamma)))  

if __name__ == '__main__':
    env = gym.make('LQG1D-v0')

    theta_star = env.computeOptimalK()[0] 
   
    action_volume = 2*env.max_action #|A|
    R = np.asscalar(env.Q*env.max_pos**2+env.R*env.max_action**2)
    M_phi = env.max_pos

    gamma = env.gamma    
    sigma = 1 
    N = int(sys.argv[1])   #batch size
    H = env.horizon
    theta = 0 #initial value
    delta = 0.2
    grad_estimator = reinforce_grad
    d = reinforce_d(R,M_phi,H,delta,sigma,gamma) #constant for variance bound

    seed = None
    verbose = 1
    record = len(sys.argv) > 2
    env.seed(seed)
    np.random.seed(seed)  
    
    if record:
        fp = open(sys.argv[2],'a')    

    i = 0 #iteration
    while True:
        i+=1 
        noises = np.random.normal(0,1,(N,H))
        if verbose > 0:
            print 'it:', i, 'theta:', theta, 'theta*:', theta_star
         
        disc_rewards = np.zeros((N,H))
        scores = np.zeros((N,H))
        for n in range(N): 
            s = env.reset()

            for l in range(H): 
                a = gauss_policy(s,theta,sigma,noises[n,l])
                scores[n,l] = gauss_score(s,a,theta,sigma)
                s,r,done,_ = env.step(a)
                disc_rewards[n,l] = gamma**l*r                
            
        grad_J = grad_estimator(scores,disc_rewards)            
            
        # adaptive step size
        c = (R*M_phi**2*(gamma*math.sqrt(2*math.pi)*sigma + 2*(1-gamma)*action_volume))/ \
            (2*(1-gamma)**3*sigma**3*math.sqrt(2*math.pi))
    
        epsilon = d/math.sqrt(N)
        #N_star = (8/(13-3*math.sqrt(17)))*d**2/grad_J**2
        if verbose > 0:
            print 'epsilon:', epsilon, 'grad:', grad_J #, 'N*:', N_star
         
        down = max(abs(grad_J) - epsilon,0)
        if down==0:
            break
        up = np.abs(grad_J) + epsilon
        
        alpha = down**2/(2*c*up**2)
        if verbose > 0:
            print 'alpha:', alpha
        
        #update
        theta+=alpha*grad_J
    
    print 'alpha=0 in',i,'iterations, theta =',theta
    if record:
        fp.write('{} {}\n'.format(i,theta))
        fp.close()
