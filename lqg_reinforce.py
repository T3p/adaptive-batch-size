import gym
from ifqi.envs.lqg1d import LQG1D
import numpy as np
from numpy.linalg import norm
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

    gamma = env.gamma    
    sigma = env.sigma_noise
    N = 10000   #batch size
    H = env.horizon
    theta = np.zeros((m))
    alpha = 0.0001 #learning rate
    #delta = 0.01
    b = 0 #baseline

    c = (R*M_phi**2*(gamma*math.sqrt(2*math.pi)*sigma + 2*(1-gamma)*action_volume))/ \
            (2*(1-gamma)**3*sigma**3*math.sqrt(2*math.pi))
    
    #d = math.sqrt((R**2*M_phi**2*H*(1-gamma**H)**2)/ \
    #                (sigma**2*(1-gamma)**2*delta)) 

    for i in range(max_iter): 
        if verbose > 0:
            print 'iteration:', i, 'theta:', theta, 'theta*:', theta_star
        #convergence
        if abs(theta[0] - theta_star[0]) <= accuracy*theta_star:
            break
        
        grad_J = np.zeros((m))
    
        for n in range(N):
            #episode
            s = env.reset()
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

        
        #adaptive step size
        alpha = norm(grad_J,2)**2/(2*c*norm(grad_J,1)**2)
        if verbose > 0:
            print 'alpha:', alpha
                      
        #update
        theta+=alpha*grad_J

