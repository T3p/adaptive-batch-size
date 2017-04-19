from dam import Dam

import math
import numpy as np
from numpy.linalg import norm
from gym.envs.registration import register

#Radial Basis Function
def rbf(c,w,s):
    return math.exp(-norm(s-c)/w)

def spaceout(n,lo,hi):
    step = (hi - lo)/(n+1)
    return [lo+k*step for k in range(1,n+1)]
        
register(
    id='DamWrap-v0',
    entry_point='dam_wrap:DamWrap',
)

#Wrapper for Dam environment
class DamWrap(Dam):
    def __init__(self,dreward=2,penalize=False):
        super(DamWrap,self).__init__()
        
        self.centers = spaceout(4,-20,190)
        self.dim = len(self.centers)+1
        self.width = 60    

        self.obj_weights = [0.5,0.5]
        assert len(self.obj_weights)==dreward
        assert sum(self.obj_weights)==1

    def step(self,action,render=False):
        s,r,done,info = super(DamWrap,self).step(action,render)
        
        #Model state: RBFs
        new_s = [1]
        for c in self.centers:
            new_s.append(rbf(c,self.width,s))
        
        #Model reward: convex combination
        new_r = np.dot(self.obj_weights,r)

        return new_s,new_r,done,info
