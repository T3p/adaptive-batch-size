from numpy.random import normal
import numpy as np

def inv(x):
    return 1./x if np.isscalar(x) else np.linalg.inv(x) 

class GaussPolicy:
    """Gaussian parametric policy with mean linear in the features and constant variance matrix"""

    def __init__(self,theta,cov):
        """Parameters:
        theta -- d X m parameter matrix, where d is the dimensionality of the action space 
            and m the size of the feature vector
        cov -- d X d covariance matrix (variance in the scalar case)
        """
        self.theta = theta 
        cov = np.asmatrix(cov)
        assert np.array_equal(cov,np.transpose(cov))
        self.cov = cov
        self.act_dim = 1 if np.isscalar(cov) else np.shape(cov)[0]
 
    def act(self,phi,noise=None):
        """Policy stochastic mapping

        Parameters:
        phi -- feature vector of state s
        noise (optional) -- gaussian noise realization

        Returns: action a sampled from pi(a|s)
        """
        if noise is None:
            noise = normal(0,1,self.act_dim)

        a = np.dot(self.theta,phi) + np.dot(self.cov,noise)
        return np.asscalar(a) if np.size(a)==1 else np.ravel(a)

    def score(self,phi,a):
        """Score function
        
        Parameters:
        phi -- feature vector of  state s
        a -- action

        Returns: the gradient w.r.t. theta of the logarithm of the policy density function pi(a|s)
        """
        phi = np.asmatrix(phi).T
        a = np.asmatrix(a).T
        score = np.dot(inv(self.cov), \
                    np.dot((a - np.dot(self.theta,phi)),np.transpose(phi)))                
        return np.asscalar(score) if np.size(score)==1 else score
