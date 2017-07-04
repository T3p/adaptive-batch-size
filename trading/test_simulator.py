from stock_simulator import StockSimulator
import gym
from stock_simulator import TradingEnv

s = env = gym.make('tradingsim-v0')
while True:
    s,r,done,info = env.step(env.action_space.sample())
    nav = info['nav']
    print s
    print nav
    print

    if done: 
        s = env.reset()
        print 'NEW'
