import gym
from ifqi.envs.lqg1d import LQG1D
import numpy as np
import math

def gauss_policy(s,theta,sigma):
    return np.random.normal(s*theta,sigma)

def gauss_score(s,a,theta,sigma):
    return (a-s*theta)*s/(sigma**2)

if __name__ == '__main__':
    env = gym.make('LQG1D-v0')

    theta_star = env.computeOptimalK()[0] 

    action_volume = 2*env.max_action
    R = -np.inf#np.asscalar(np.dot(env.max_pos,
                      #np.dot(env.Q, env.max_pos)) + \
        #np.dot(env.max_action, np.dot(env.R, env.max_action)))
    M_phi = env.max_pos

    gamma = 0.95#env.gamma    
    sigma = 10 
    N = 10000   #batch size
    H = 20 #env.horizon
    verbose = 1
    theta = 0
    delta = 0.1
    
    i = 0
    while True:
        i+=1 
        if verbose > 0:
            print 'iteration:', i, 'theta:', theta, 'theta*:', theta_star
        
        grad_J = 0
        score_H = []
        q_H = [] 
        for n in range(N): 
            #episode
            s = env.reset()
            
            score = 0  
            q = 0
            for l in range(H): 
                a = gauss_policy(s,theta,sigma)
                score+=gauss_score(s,a,theta,sigma)
                    
                s,r,done,_ = env.step(a)
                q+= gamma**l*r
 
                if abs(r) > R:
                    R = abs(r)  
               
            score_H.append(score)
            q_H.append(q)
        
        b = np.mean(map(lambda x,y: x**2*y, score_H,q_H)) / \
                np.mean(map(lambda x: x**2, score_H))

        grad_J = np.mean(map(lambda x,y: x*(y-b), score_H, q_H))
            
        # adaptive step size
        c = (R*M_phi**2*(gamma*math.sqrt(2*math.pi)*sigma + 2*(1-gamma)*action_volume))/ \
            (2*(1-gamma)**3*sigma**3*math.sqrt(2*math.pi))
    
        d = math.sqrt((R**2*M_phi**2*H*(1-gamma**H)**2)/ \
                    (sigma**2*(1-gamma)**2*delta)) 
        epsilon = d/math.sqrt(N)
        print 'epsilon:', epsilon, 'grad:', grad_J
        
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
