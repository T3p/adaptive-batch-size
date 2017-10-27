import numpy as np
from utils import *

"""Policy gradient estimation algorithms"""

   
def reinforce(features,actions,rewards,gamma,pol,use_baseline=True,average=True):
    """Batch REINFORCE policy gradient estimator with variance-minimizing baseline

    Parameters: 
    features -- N x H x m array containing feature vectors for N episodes of length H
    actions -- N x H x d array containing actions for N episodes of length H
    rewards -- N x H array containing rewards for N episodes of length H 
    gamma -- discount factor
    use_baseline -- if False, a baseline of b=0 is used instead

    Returns:
    the averaged gradient estimate if average==True, an array containing the N estimates otherwise
    """
    
    #Data 
    assert features.shape[:2]==actions.shape[:2]==rewards.shape[:2]
    N = features.shape[0]
    H = features.shape[1]
   
    #
    discounts = gamma**np.indices((N,H))[1]
    disc_rewards = rewards*discounts 
    q = np.sum(disc_rewards,1)
    
    #Eligibility vector
    scores = apply_along_axis2(pol.score,2,actions,features)
    sum_of_scores = np.sum(scores,1)
    
    #Optimal baseline
    b = 0
    if use_baseline and N>1:
        b = np.mean(sum_of_scores**2*q[:,np.newaxis])/np.mean(sum_of_scores**2)
    
    #Gradient
    estimates = sum_of_scores*(q-b)[:,np.newaxis]
    return np.mean(estimates) if average else estimates


def gpomdp(features,actions,rewards,gamma,pol,use_baseline=True,average=True):
    """Batch G(PO)MDP policy gradient estimator with variance-minimizing baseline

    Parameters:  
    features -- N x H x m array containing feature vectors for N episodes of length H
    actions -- N x H x d array containing actions for N episodes of length H
    rewards -- N x H array containing rewards for N episodes of length H
    gamma -- discount factor
    use_baseline -- if False, a baseline of b=0 is used instead

    Returns:
    the averaged gradient estimate if average==True, an array containing the N estimates otherwise
    """
    #Data 
    assert features.shape[:2]==actions.shape[:2]==rewards.shape[:2]
    N = features.shape[0]
    H = features.shape[1] 

    #Q function
    discounts = gamma**np.indices((N,H))[1]
    disc_rewards = rewards*discounts 

    #Eligibility vector    
    scores = apply_along_axis2(pol.score,2,actions,features)
    cum_scores = np.cumsum(scores,1)

    #Optimal baseline:
    b = np.zeros(H)
    if use_baseline and N>1:
        den = np.mean(cum_scores**2,0)
        den[den==0] = 1
        b = np.mean(cum_scores**2*disc_rewards,0)/den
        
    #gradient estimate:
    estimates =  np.sum(cum_scores*(disc_rewards - b),1)
    return np.mean(estimates) if average else estimates


