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
    m = features.shape[2] if len(features.shape)>2 else 1

    #Q function 
    disc_rewards = __discount(rewards,gamma)
    q = np.sum(disc_rewards,1)
    
    #Eligibility vector
    scores = apply_along_axis2(pol.score,2,actions,features)
    sum_of_scores = np.sum(scores,1)
    
    #Optimal baseline
    
    b = np.zeros(m)
    if use_baseline and N>1: 
        den = np.asarray(np.mean(sum_of_scores**2,0))
        np.putmask(den,den==0,1)
        b = np.mean(((sum_of_scores**2).T*q).T,0)/den
  
    #Gradient
    estimates = (sum_of_scores.T*q).T - sum_of_scores*b
   
    return np.mean(estimates,0) if average else estimates


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
    m = features.shape[2] if len(features.shape)>2 else 1

    #Q function
    disc_rewards = __discount(rewards,gamma)

    #Eligibility vector    
    scores = apply_along_axis2(pol.score,2,actions,features)
    cum_scores = np.cumsum(scores,1)

    #Optimal baseline:
    b = np.zeros((H,m))
    if use_baseline and N>1:
        den = np.mean(cum_scores**2,0)
        np.putmask(den,den==0,1)
        b = np.mean(((cum_scores**2).T*disc_rewards.T).T,0)/den
        
    #gradient estimate:
    estimates =  np.sum((cum_scores.T*disc_rewards.T).T - cum_scores*b,1)
    return np.mean(estimates,0) if average else estimates

def performance(rewards,gamma=None,average=True):
    discounted = (gamma!=None)
    if discounted:
        Js = np.sum(__discount(rewards,gamma),1)
    else:
        Js = np.sum(rewards,1)

    return np.mean(Js) if average else Js

def __discount(rewards,gamma):
    #Applies the discount factor to rewards
    N = rewards.shape[0]
    H = rewards.shape[1] 
    discounts = gamma**np.indices((N,H))[1]
    return rewards*discounts
