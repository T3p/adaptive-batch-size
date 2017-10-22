import numpy as np
from gradient_estimation import *
from policies import GaussPolicy

h = np.array([[[0,1,0],[1,0,1],[1,1,1],[0,0,0]],[[0,0,0],[0,1,0],[1,0,1],[1,0,1]]])

pol = GaussPolicy(0,1)
assert reinforce(h,0.5,pol)==0
assert gpomdp(h,0.5,pol)==0

pol = GaussPolicy(2,0.0001)
assert pol.act(3,noise=0)==6
assert pol.score(3,8)==60000

pol = GaussPolicy([[2,0,0],[4,2,0]],[[0.0001,0],[0,0.1]])
assert np.array_equal(pol.act([3,6,9],noise=[0,0]),[6,24])
assert np.array_equal(pol.score([3,6,9],[8,10]),[[60000,120000,180000],[-420,-840,-1260]])

