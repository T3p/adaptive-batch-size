import numpy as np

def reinforce(history,gamma,pol,use_baseline=True,average=True):
    """Batch REINFORCE policy gradient estimator with variance-minimizing baseline

    Parameters: 
    history -- N*H*3 array containing (in this order) states, actions and rewards of N episodes 
                of length H
    gamma -- discount factor
    use_baseline -- if False, a baseline of b=0 is used instead

    Returns:
    the averaged gradient estimate if average==True, an array containing the N estimates otherwise
    """
    
    #Data
    assert history.shape[2] == 3
    N = history.shape[0]
    H = history.shape[1]
    states = history[:,:,0]
    actions = history[:,:,1]
    rewards = history[:,:,2]
   
    #Q function
    discounts = gamma**np.indices((N,H))[1]
    disc_rewards = rewards*discounts 
    q = np.sum(disc_rewards,1)
    
    #Eligibility vector
    scores = np.array([[pol.score(states[i,j],actions[i,j]) for j in range(H)] for i in range(N)])
    sum_of_scores = np.sum(scores,1)
    
    #Optimal baseline
    b = 0
    if use_baseline and N>1:
        b = np.mean(sum_of_scores**2*q)/np.mean(sum_of_scores**2)
    
    #Gradient
    estimates = sum_of_scores*(q-b)
    return np.mean(estimates) if average else estimates




def gpomdp(history,gamma,pol,use_baseline=True,average=True):
    """Batch G(PO)MDP policy gradient estimator with variance-minimizing baseline

    Parameters: 
    history -- N*H*3 array containing (in this order) states, actions and rewards of N episodes 
                of length H
    gamma -- discount factor
    use_baseline -- if False, a baseline of b=0 is used instead

    Returns:
    the averaged gradient estimate if average==True, an array containing the N estimates otherwise
    """

    #Data
    assert history.shape[2] == 3
    N = history.shape[0]
    H = history.shape[1]
    states = history[:,:,0]
    actions = history[:,:,1]
    rewards = history[:,:,2]
   
    #Q function
    discounts = gamma**np.indices((N,H))[1]
    disc_rewards = rewards*discounts 

    #Eligibility vector    
    scores = np.array([[pol.score(states[i,j],actions[i,j]) for j in range(H)] for i in range(N)])
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



