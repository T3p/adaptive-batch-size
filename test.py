import numpy as np
from gradient_estimation import *
from policies import GaussPolicy

s = np.array([[0,1,1,0],[0,0,1,1]])
a = np.array([[1,0,1,0],[0,1,0,0]])
r = np.array([[0,1,1,0],[0,0,1,1]])
pol = GaussPolicy(0,1)
assert reinforce(s,a,r,0.5,pol)==0
assert gpomdp(s,a,r,0.5,pol)==0

pol = GaussPolicy(2,0.0001)
assert pol.act(3,noise=0)==6
assert pol.score(3,8)==60000

pol = GaussPolicy([2,0,0,4,2,0],[[0.0001,0],[0,0.1]])
assert np.array_equal(pol.act([3,6,9],noise=[0,0]),[6,24])
assert np.array_equal(pol.score([3,6,9],[8,10]),[60000,120000,180000,-420,-840,-1260])

pol = GaussPolicy([1,0,0,1],[[0.1,0],[0,0.1]])
s = np.array([[[0,0],[1,0]],[[0,0],[0,1]]])
a = np.array([[[1,0],[1,1]],[[0,1],[1,1]]])
r = np.array([[0,1],[0,1]])
assert reinforce(s,a,r,0.9,pol)==0
