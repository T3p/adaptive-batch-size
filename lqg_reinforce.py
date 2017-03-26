import gym
from ifqi.envs.lqg1d import LQG1D
import numpy as np
import math

def gauss_policy(phi,theta,sigma):
    mu = np.dot(phi,theta)
    return np.random.normal(mu,sigma)

def gauss_score(phi,a,theta,sigma):
    mu = np.dot(phi,theta)
    return (a-mu)*phi/(sigma**2)

if __name__ == '__main__':
    env = gym.make('LQG1D-v0')
    
    #s in [-10,10], a in [-8,8]
    m = 1 #feature space dimensions
    theta_star = env.computeOptimalK()[0] 
    accuracy = 0.01
    max_iter = 30000
    do_render = False    
    verbose = 1

    action_volume = 2*env.max_action
    R = np.asscalar(np.dot(env.max_pos,
                      np.dot(env.Q, env.max_pos)) + \
        np.dot(env.max_action, np.dot(env.R, env.max_action)))
    M_phi = env.max_pos
    print action_volume, R, M_phi

    gamma = env.gamma    
    sigma = 0.1
    N = 10000   #batch size
    H = 20      #episode length
    theta = np.zeros((m))
    alpha = 0.0001 #learning rate
    b = 0 #baseline
 
    for i in range(max_iter): 
        if verbose > 0:
            print 'iteration:', i, 'theta:', theta, 'theta*:', theta_star
        #convergence
        if abs(theta[0] - theta_star[0]) <= accuracy*theta_star:
            break
        
        grad_J = np.zeros((m))

        #batch
        for _ in range(N):
            #episode
            env.reset()
            s = env.get_state()
            states = []
            actions = []  
            g = 0

            for l in range(H):
                a = gauss_policy(s,theta,sigma)
                states.append(s)
                actions.append(a)

                s,r,_,_ = env.step(a) 
                g+= (gamma**l)*r - b 
                
                if do_render:
                    env.render()
            
            for k in range(H):
                grad_J += gauss_score(states[k],actions[k],theta,sigma) * g
        grad_J/=N
        
        #update
        theta+=alpha*grad_J

