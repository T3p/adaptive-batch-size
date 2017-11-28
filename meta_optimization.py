import math
from utils import *
from gradient_estimation import Estimator

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
        self.diameter = diameter


class GradStats:
    """Statistics about a gradient estimate"""
  
    def __init__(self,grad_samples):
        """Parameters:
            grad_samples: gradient estimates
        """
        self.grad = np.mean(grad_samples,0)
        self.max_grad = np.max(abs(self.grad))
        self.k_max = np.argmax(abs(self.grad))
        self.grad_samples = grad_samples[:,self.k_max] if len(grad_samples.shape)>1 else grad_samples
        self.sample_range = self.sample_var = None

    def get_estimate(self):
        return self.grad

    def get_max(self):
        return self.max_grad

    def get_amax(self):
        return self.k_max

    def get_var(self):
        if self.sample_var==None:
            self.sample_var = np.var(self.grad_samples,ddof=1)
        return self.sample_var

    def get_range(self):
        if self.sample_range==None:
            self.sample_range = max(self.grad_samples) - min(self.grad_samples)
        return self.sample_range
        

class OptConstr:
    """Constraints on the meta-optimization process"""

    def __init__(self,delta=0.95,N_min=2,N_max=999999,N_tot=30000000):
        """Parameters:
            delta : maximum allowed worsening probability
            N_min : min allowed batch size
            N_max : max allowed batch size
            N_tot : total number of possible trajectories
        """

        self.delta = delta
        self.N_min = N_min
        self.N_max = N_max
        self.N_tot = N_tot

#Default constraints
default_constr = OptConstr()


def alphaStar(pol,tp):
    """Optimal step size for the adaBatch algorithm when the corresponding optimal
        batch size is used

        Parameters:
        pol -- policy in use
        tp -- TaskProp object containing the (true or last estimated) properties of the task
   """

    c = pol.penaltyCoeff(tp.R,tp.M,tp.gamma,tp.volume)
    return (13-3*math.sqrt(17))/(4*c)

def alphaPost(pol,tp,max_grad,eps):
    """Optimal step size given an upper bound of the estimaton error,
        depending on the batch size that is actually used
        
        Parameters:
        pol -- policy in use
        tp -- TaskProp object containing the (true or last estimated) properties of the task
        gs -- GradStats object containing statistics on the last gradient estimate
    """
    c = pol.penaltyCoeff(tp.R,tp.M,tp.gamma,tp.volume)
    return (max_grad - eps)**2/(2*c*(max_grad + eps)**2)

def gradRange(pol,tp):
    """Range of the gradient estimate
        
        Parameters:
        pol -- policy in use
        tp -- TaskProp object containing the (true or last estimated) properties of the task
    """
    Q_sup = float(tp.volume*tp.R)/(1-tp.gamma)
    return float(tp.M*tp.diameter*Q_sup)/pol.sigma**2

def estError(d,f,N):
    """Generic estimation error bound

        Parameters:
        d -- 1/sqrt(N) coefficient
        f -- 1/N coefficient
        N -- batch size 
    """
    return float(d)/math.sqrt(N) + float(f)/N


class metaSelector:
    def __init__(self,alpha,N):
        self.alpha = alpha
        self.N = N

    def select(self,pol,gs,tp,N_pre):
        return self.alpha,self.N,False


class MetaOptimizer(metaSelector):
    """Tool to compute the optimal meta-parameters for a policy gradient problem"""

    def __init__(self,bound_name='bernstein',constr=default_constr,estimator='gpomdp',samp=True):

        
        bounds = {'chebyshev': self.__chebyshev, 'hoeffding': self.__hoeffding, 'bernstein': self.__bernstein}

        self.bound = bounds[bound_name]
        self.constr = constr
        self.estimator = estimator
        self.samp = samp

    def select(self,pol,gs,tp,N_pre):
        """Compute optimal step size and batch size

            Parameters:
            pol -- policy in use
            gs -- GradStats object containing statistics about last gradient estimate
            tp -- TaskProp object containing the (true or last estimated) properties of the task
            N_pre -- batch size that was actually used to compute the last gradient estimate
            
            Returns:
            alpha -- the optimal non-scalar step size
            N -- the optimal batch size
            unsafe -- true iff no improvement can be guaranteed at all
        """
        d,f,eps_star,N_star = self.bound(pol,gs,tp)
        actual_eps = estError(d,f,N_pre)
        
        alpha_k = alphaPost(pol,tp,gs.get_max(),actual_eps)
        N = min(self.constr.N_max,max(self.constr.N_min,N_star))
        safe = eps_star<gs.get_max()

        alpha = np.zeros(pol.param_len)
        alpha[gs.get_amax()] = alpha_k

        return alpha,N,safe 

    def __closedOpt(self,d,max_grad):
        #Generic closed form optimization for N and corresponding estimation error

        eps_star = 0.25*(math.sqrt(17) - 3)*max_grad
        N_star = int(math.ceil(float(d**2)/eps_star**2))
        return eps_star,N_star


    def __chebyshev(self,pol,gs,tp):
        #Batch size optimizer using Chebyshev's bound
        if self.estimator=='reinforce':
            d =  math.sqrt((tp.R**2*tp.M**2*tp.H*(1-tp.gamma**tp.H)**2)/ \
                    (pol.sigma**2*(1-tp.gamma)**2*self.constr.delta))
        elif self.estimator=='gpomdp':
            d = math.sqrt((tp.R**2*tp.M**2)/(self.constr.delta*pol.sigma**2*(1-tp.gamma)**2) * \
                           ((1-tp.gamma**(2*tp.H))/(1-tp.gamma**2)+ tp.H*tp.gamma**(2*tp.H)  - \
                                2 * tp.gamma**tp.H  * (1-tp.gamma**tp.H)/(1-tp.gamma)))
        else:
            assert False

        return (d,0) + self.__closedOpt(d,gs.get_max())

    def __hoeffding(self,pol,gs,tp):
        #Batch size optimizer using Hoeffding's bound
        if self.samp:
            rng = gs.get_range()
        else:
            rng = gradRange(pol,tp)

        d = rng*math.sqrt(math.log(2./self.constr.delta)/2)
        return (d,0) + self.__closedOpt(d,gs.get_max())


    def __evaluateN(self,N,d,f,c,max_grad):
        #Objective function Upsilon for batch size N
        eps = float(d)/math.sqrt(N) + float(f)/N
        upsilon = (max_grad - eps)**4/ \
                    (4*c*(max_grad + eps)**2*N)
        return upsilon,eps

    def __bernstein(self,pol,gs,tp):
        #Batch size optimizer using an empirical Bernstein's bound (Mnih et al., 2008)
        if self.samp:
            rng = gs.get_range()
        else:
            rng = gradRange(pol,tp)

        c = pol.penaltyCoeff(tp.R,tp.M,tp.gamma,tp.volume)
        d = math.sqrt(2*math.log(3.0/self.constr.delta)*gs.get_var())
        f = 3*rng*math.log(3.0/self.constr.delta)

        N_0 = min(self.constr.N_max,max(self.constr.N_min,int(((d + math.sqrt(d**2 + 4*f*gs.get_max())) \
                /(2*gs.get_max()))**2) + 1))
        ups_max = -np.inf
        eps_star = np.inf
        N_star = N_0
        for n in range(N_0,self.constr.N_max):
            ups,eps = self.__evaluateN(n,d,f,c,gs.get_max())
            if ups>ups_max:
                ups_max = ups
                eps_star = eps
                N_star = n
            else:
                break

        return d,f,eps_star,N_star


