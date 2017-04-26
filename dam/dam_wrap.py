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
        self.centers = spaceout(2,-20,190)
        self.dim = len(self.centers)+1
        self.width = 60    
        
        super(DamWrap,self).__init__()
        
        self.obj_weights = [0.5,0.5]
        assert len(self.obj_weights)==dreward
        assert sum(self.obj_weights)==1

    def features(self,s):
        features = [1]
        for c in self.centers:
            features.append(rbf(c,self.width,s))
        return features

    def step(self,action,noise,render=False):
        s,r,done,info = super(DamWrap,self).step(action,render,noise)
        
        #Model state: RBFs
        new_s = self.features(s)       

        #Model reward: convex combination
        new_r = np.dot(self.obj_weights,r)

        return np.array(new_s),new_r,done,info

    def reset(self,state=None,scalar=False):
        super(DamWrap,self).reset(state)
        if not scalar:
            return np.array(self.features(self.get_state()))
        else:
            return self.get_state()
