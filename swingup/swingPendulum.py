# -*- coding: utf-8 -*-
"""
Created on Fri Sep  9 09:25:48 2016

@author: samuele

Pendulum as described in Reinforcement Learning in Continuous Time and Space
"""

import gym
import numpy as np
from gym import spaces
from gym.utils import seeding

import ifqi.utils.spaces as fqispaces
from os import path

#classic_control
from gym.envs.registration import register
register(
    id='swing-v0',
    entry_point='swingPendulum:SwingPendulum'
)

class SwingPendulum(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 15
    }

    def __init__(self, **kwargs):
        self.horizon = 100
        self.gamma = 0.9

        self._m = 1.
        self._l = 1.
        self._g = 9.8
        self._mu = 0.01
        self._dt = 0.02

        # gym attributes
        self.viewer = None
        high = np.array([np.inf, np.inf])
        self.observation_space = spaces.Box(low=-high, high=high)
        self.action_space = spaces.Box(low=-5.,high=5.,shape=(1,))

        # initialize state
        self.seed()
        self.reset()

    def _step(self, action, render=False):
        u = np.clip(action[0],self.action_space.low,self.action_space.high)
        self.last_u = u #for rendering
        theta, theta_dot = tuple(self.get_state())

        theta_ddot = (-self._mu*theta_dot + self._m*self._g*self._l*np.sin(theta) + u)/ \
                        (self._m*self._l**2)

     
        # bound theta_dot
        theta_dot_temp = theta_dot + self._dt*theta_ddot
        if theta_dot_temp > np.pi / self._dt:
            theta_dot_temp = np.pi / self._dt
        if theta_dot_temp < -np.pi / self._dt:
            theta_dot_temp = -np.pi / self._dt

        theta_dot = theta_dot_temp
        theta += theta_dot * self._dt

        # adjust Theta
        if theta > np.pi:
            theta -= 2 * np.pi
        if theta < -np.pi:
            theta += 2 * np.pi

        self._state = np.array([theta, theta_dot])
        reward = np.cos(theta)

        return self.get_state(), reward, False, {}

    def reset(self, state=None):
        if state is None:
            theta = self.np_random.uniform(low=-np.pi, high=np.pi)
            self._state = np.array([theta, 0.])
        else:
            self._state = np.array(state)

        return self.get_state()

    def get_state(self):
        return self._state

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]    

    def _render(self, mode='human', close=False):
        if close:
            if self.viewer is not None:
                self.viewer.close()
                self.viewer = None
            return

        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(500,500)
            self.viewer.set_bounds(-2.2,2.2,-2.2,2.2)
            rod = rendering.make_capsule(1, .2)
            rod.set_color(.8, .3, .3)
            self.pole_transform = rendering.Transform()
            rod.add_attr(self.pole_transform)
            self.viewer.add_geom(rod)
            axle = rendering.make_circle(.05)
            axle.set_color(0,0,0)
            self.viewer.add_geom(axle)
            fname = path.join(path.dirname(__file__), "assets/clockwise.png")
            self.img = rendering.Image(fname, 1., 1.)
            self.imgtrans = rendering.Transform()
            self.img.add_attr(self.imgtrans)

        self.viewer.add_onetime(self.img)
        self.pole_transform.set_rotation(self._state[0] + np.pi/2)
        if self.last_u:
            self.imgtrans.scale = (-self.last_u/2, np.abs(self.last_u)/2)

        return self.viewer.render(return_rgb_array = mode=='rgb_array')
