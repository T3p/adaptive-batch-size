import math
import numpy as np
from collections import deque
import gym
from gym import spaces

class StockSimulator:
    
    def __init__(self,z_0=0,b_0=0,k=3,alpha=0.9,window=1):
        self.z_0 = z_0
        self.b_0 = b_0
        self.k = k
        self.alpha = alpha
        self.window = window
        self.history = deque([0]*self.window)
        self.reset()

    def reset(self):
        self.z = z = self.z_0
        self.b = self.b_0
        self.z_max = self.z_min = z
        return np.array(self.history)

    def step(self):
        #Random walk
        eps = np.random.normal()
        ni = np.random.normal()#*0.01
        self.z = z = self.z + self.b + self.k*eps
        self.b = self.alpha*self.b + ni
        
        #Synthetic price series
        self.z_max = z_max = max(self.z_max,z)
        self.z_min = z_min = min(self.z_min,z)
        Z = math.exp(z/(z_max - z_min))
        self.history.popleft()
        self.history.append(Z)
        return np.array(self.history)
        

from gym.envs.registration import register
register(
    id = 'tradingsim-v0',
    entry_point = 'stock_simulator:TradingEnv'
)

class TradingEnv(gym.Env):
    def __init__(self):
        self.days = 10
        self.sim = StockSimulator(window=1)
        self.action_space = spaces.Box(low=-1,high=1,shape=(1,))

        self.trade_cost = 0.1  
        self.time_cost = 0
    
        self.reset()

    def step(self,action):
        assert self.action_space.contains(action)
        old_value = self.sim.history[-1]
        observation = self.sim.step()
        self.steps+=1
        done = self.steps >=self.days
        
        #Market
        if old_value==0:
            retrn = 0
        else:
            retrn = (observation[-1] - old_value)/old_value 
        self.trades = trades = self.posn - action
        self.posn = posn = action
        cost = trades*self.trade_cost + self.time_cost 
        self.nav*=(1 + self.reward)
        self.reward = reward = posn*retrn - cost
        info = {'nav': self.nav}

        return observation,reward,done,info
        
    def reset(self):
        self.steps = 0
        observation = self.sim.step()#self.sim.reset()
        self.posn = 0
        self.nav = 1
        self.reward = 0
        return observation
