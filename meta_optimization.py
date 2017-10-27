import math
from utils import *
from gradient_estimation import reinforce,gpomdp

class TaskProp:
    """Properties of the RL task, true or estimated from experience"""

    def __init__(self,R,M,gamma,H,min_state,max_state,min_action,max_action,volume,diameter=None):
        """Parameters:
            R -- max absolute-value reward
            M -- upper bound on all state features
            gamma -- discount factor
            H -- episode length
            min_state --
            max_state --
            min_action --
            max_action --
            volume -- volume of the action space
            diameter -- maximum euclidean distance among possible actions
        """
        self.R = R
        self.M = M
        self.gamma = gamma
        self.H = H
        self.diameter = diameter
        self.min_state = min_state
        self.max_state = max_state
        self.min_action = min_action
        self.max_action = max_action
        self.volume = volume


class GradStats:
    """Statistics about the last gradient estimate"""
  
    def __init__(self,max_grad,sample_var=None,sample_range=None):
        """Parameters:
            max_grad -- infinity norm of the gradient estimate
            sample_var -- sample variance of the gradient estimator
            sample_range -- estimated range of the gradient estimator
        """
        self.max_grad = max_grad
        self.sample_var = sample_var 
        self.sample_range = sample_range        


class OptConstr:
    """Constraints on the meta-optimization process"""

    def __init__(self,N_min,N_max):
        """Parameters:
            N_min : min allowed batch size
            N_max : max allowed batch size
        """
        self.N_min = N_min
        self.N_max = N_max

def alphaStar(pol,t):
    """Optimal step size for the adaBatch algorithm when the corresponding optimal
        batch size is used

        Parameters:
        pol -- policy in use
        t -- TaskProp object containing the (true or last estimated) properties of the task
    """

    c = pol.penaltyCoeff(t.R,t.M,t.gamma,t.volume)
    return (13-3*math.sqrt(17))/(4*c)

def alphaPost(pol,t,max_grad,eps):
    """Optimal step size given an upper bound of the estimaton error,
        whatever batch size is actually used
        
        Parameters:
        pol -- policy in use
        t -- TaskProp object containing the (true or last estimated) properties of the task
        s -- GradStats object containing statistics on the last gradient estimate
    """
    c = pol.penaltyCoeff(t.R,t.M,t.gamma,t.volume)
    return (max_grad - eps)**2/(2*c*(max_grad + eps)**2)

def __gradRange(pol,t):
    #Upper bound on the range of the gradient estimate

    Q_sup = float(t.volume*t.R)/(1-t.gamma)
    return float(t.M*t.diameter*Q_sup)/pol.sigma**2

def __closedOpt(d,max_grad):
    #Generic closed form optimization for N and corresponding estimation error

    eps_star = 0.25*(math.sqrt(17) - 3)*max_grad
    N_star = int(math.ceil(float(d**2)/eps_star**2))
    return eps_star,N_star


def chebyshev(delta,pol,s,t,con=None,estimator=gpomdp,samp=False):
    if estimator==reinforce:
        d =  math.sqrt((t.R**2*t.M**2*t.H*(1-t.gamma**t.H)**2)/ \
                (pol.sigma**2*(1-t.gamma)**2*delta))
    elif estimator==gpomdp:
        d = math.sqrt((t.R**2*t.M**2)/(delta*pol.sigma**2*(1-t.gamma)**2) * \
                       ((1-t.gamma**(2*t.H))/(1-t.gamma**2)+ t.H*t.gamma**(2*t.H)  - \
                            2 * t.gamma**t.H  * (1-t.gamma**t.H)/(1-t.gamma)))
    else:
        assert False

    return (d,0) + __closedOpt(d,s.max_grad)

def hoeffding(delta,pol,s,t,con=None,estimator=gpomdp,samp=False):
    if samp:
        rng = s.sample_range
    else:
        rng = __gradRange(pol,t)

    d = rng*math.sqrt(math.log(2./delta)/2)
    return (d,0) + __closedOpt(d,s.max_grad)


def __evaluateN(N,d,f,c,max_grad):
    #Objective function Upsilon for batch size N
    eps = float(d)/math.sqrt(N) + float(f)/N
    upsilon = (max_grad - eps)**4/ \
                (4*c*(max_grad + eps)**2*N)
    return upsilon,eps

def bernstein(delta,pol,s,t,con,estimator=gpomdp,samp=False):
    if samp:
        rng = s.sample_range
    else:
        rng = __gradRange(pol,t)

    c = pol.penaltyCoeff(t.R,t.M,t.gamma,t.volume)
    d = math.sqrt(2*math.log(3.0/delta)*s.sample_var)
    f = 3*rng*math.log(3.0/delta)

    N_0 = min(con.N_max,max(con.N_min,int(((d + math.sqrt(d**2 + 4*f*s.max_grad)) \
            /(2*s.max_grad))**2) + 1))
    ups_max = -np.inf
    eps_star = np.inf
    N_star = N_0
    for n in range(N_0,con.N_max):
        ups,eps = __evaluateN(n,d,f,c,s.max_grad)
        if ups>ups_max:
            ups_max = ups
            eps_star = eps
            N_star = n
        else:
            break

    return d,f,eps_star,N_star


def __estError(d,f,N):
    return float(d)/math.sqrt(N) + float(f)/N

def metaOptimize(delta,pol,s,t,con,bound,N_pre,estimator=gpomdp,samp=False):
    
    d,f,eps_star,N_star = bound(delta,pol,s,t,con,estimator,samp)
    actual_eps = __estError(d,f,N_pre)
    
    alpha = alphaPost(pol,t,s.max_grad,actual_eps)
    N = min(con.N_max,max(con.N_min,N_star))
    unsafe = eps_star>=s.max_grad

    return alpha,N,unsafe
    
