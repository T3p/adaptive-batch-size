import numpy as np
import math

#OpenAI
import gym
from lqgnd import LQGND

#parallelism
import sys
from joblib import Parallel, delayed
import multiprocessing
import tempfile, os

#meta
import time
import matplotlib.pyplot as plt

#Gaussian policy
def gauss_policy(s,theta,sigma,noise):
    return np.dot(theta,s) + np.dot(sigma,noise)

   
if __name__ == '__main__':
    #theta = -0.58*np.eye(2)
    theta = np.array([[-2.10708108e-01, 6.69330475e-02],  [-6.68543098e-02, -2.10779032e-01]])
    #theta = np.array([[-6.00422807e-01, -1.11741208e-04], [1.04054320e-04, -5.99887257e-01]])    
    #theta = np.array([[-9.55160159e-04  ,-1.19576257e-04],   [2.64307272e-04  ,-9.75963053e-04]])

    def vecf(theta):
        X,Y = np.meshgrid(np.arange(-2,2,.2),np.arange(-2,2,.2))
        
        w = h = X.shape[0]
        U = np.zeros((w,h))
        V = np.zeros((w,h))

        for i in range(w):
            for j in range(h):
                s = np.array([X[i,j],Y[i,j]])
                U[i,j],V[i,j] = gauss_policy(s,theta,0,0)

        plt.figure()
        plt.title('Predicted action')
        Q = plt.quiver(X, Y, U, V, units='width')
        #qk = plt.quiverkey(Q, 0.9, 0.9, 2, r'$2 \frac{m}{s}$', labelpos='E',
         #                  coordinates='figure')
    
    
    vecf(theta)
    theta = -0.58*np.eye(2)    
    vecf(theta)
    plt.show()
