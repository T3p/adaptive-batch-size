import numpy as np
import math
import adabatch

from gradient_estimation import *
from policies import GaussPolicy
from meta_optimization import *

"""Warning: these are not complete tests, just sanity checks"""

#gradient_estimation
s = np.array([[0,1,1,0],[0,0,1,1]])
a = np.array([[1,0,1,0],[0,1,0,0]])
r = np.array([[0,1,1,0],[0,0,1,1]])
pol = GaussPolicy(0,1)
assert reinforce(s,a,r,0.5,pol)==0
assert gpomdp(s,a,r,0.5,pol)==0
assert performance(r)==2
assert performance(r,0.5)==0.5625

#policies.GaussPolicy
pol = GaussPolicy(2,0.01)
assert pol.param_len==1
assert pol.act(3,deterministic=True)==6
assert pol.score(8,3)==600
assert abs(pol.prob(6,3)-(2*math.pi*0.01)**(-0.5)) < 1e-5
pol.act(3)
assert abs(pol.penaltyCoeff(5,2,0.9,4) - 4091538.24321) < 1e-5

pol = GaussPolicy([2,0,0,4,2,0],[[0.0001,0],[0,0.1]])
assert pol.param_len==6
assert np.array_equal(pol.act([3,6,9],deterministic=True),[6,24])
assert np.array_equal(pol.score([8,10],[3,6,9]),[60000,120000,180000,-420,-840,-1260])
assert abs(pol.prob([6,24],[3,6,9]) -  1.0/(2*math.pi*math.sqrt(0.00001))) < 1e-5
pol.act([3,6,9])
pol.penaltyCoeff(5,2,0.9,16)

pol = GaussPolicy([1,0,0,1],[[0.1,0],[0,0.1]])
assert pol.param_len==4
s = np.array([[[0,0],[1,0]],[[0,0],[0,1]]])
a = np.array([[[1,0],[1,1]],[[0,1],[1,1]]])
r = np.array([[0,1],[0,1]])
assert reinforce(s,a,r,0.9,pol).all()==0
assert gpomdp(s,a,r,0.9,pol).all()==0



#meta_optimization
pol = GaussPolicy(-1,1)
tp = TaskProp(
    R = 4,
    M = 2,
    gamma = 0.9,
    H = 20,
    min_state = -2,
    max_state = 2,
    min_action = -1,
    max_action = 1,
    volume = 2,
    diameter = 2,
    )
gs = GradStats(np.array([10,-40,60]))
con = OptConstr(
    delta = 0.95,
    N_min = 2,
    N_max = 100000
)

assert gs.get_estimate()==10
assert gs.get_range()==100
assert gs.get_var()==2500

meta_optimizers = [metaOptimizer('chebyshev',con,estimator=reinforce), 
                    metaOptimizer('chebyshev',con),
                    metaOptimizer('hoeffding',con,samp=False),
                    metaOptimizer('hoeffding',con),
                    metaOptimizer('bernstein',con,samp=False),
                    metaOptimizer('bernstein',con)]
for mo in meta_optimizers:
    print mo.select(pol,gs,tp,N_pre=300)

print
print "---- TESTS COMPLETED ----"
