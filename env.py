import gym
import numpy as np
from jax import numpy as jnp


class EuropeanCallDiscreteEnv(gym.Env):
  def __init__(self, S0=100,strike_price=100, n_steps=30, epsilon=0, sigma=0.2, risk_free=0):
    self.S0 = S0
    self.strike_price = strike_price
    self.n_steps = n_steps
    self.epsilon = epsilon
    self.sigma = sigma
    self.risk_free = risk_free

    self.S = 0
    self.day_count = 0
    self.delta = 0

    self.action_space = gym.spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32)
    self.observation_space = gym.spaces.Box(low=np.array([0, 0]), high=np.array([np.inf, 1.0]), dtype=np.float32) 
    
  def step(self, action):
    if self.day_count == self.n_steps:
      reward = self.S*self.delta*self.epsilon  
      reward -= jnp.maximum(self.S - self.strike_price, 0)
      done = True
    else:
        reward = self.S * ( self.delta - action)
        reward -= self.S * jnp.abs(self.delta-action)*self.epsilon
        rand = np.random.standard_normal()
        rand = 2*int(rand>0)-1
        self.S = self.S + rand
        self.delta = action
        self.day_count += 1
        done = False
    tao = 1 - self.day_count/365  # Time to maturity in unit of year.
    return np.array([self.S,tao]), reward, done, {}

  def reset(self):
    self.day_count = 0
    self.S = self.S0
    tao =  1 - self.day_count/365
    return np.array([self.S,tao])