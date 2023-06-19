from collections import namedtuple
from collections import deque

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

import numpy as np
import numpy.random as nr
from numpy.random import binomial
from numpy.random import choice

import gym
import datetime
from typing import Tuple
from scipy import linalg


Transitions = namedtuple('Transitions', ['obs', 'action', 'reward', 'next_obs', 'done'])

class ReplayBuffer:
    def __init__(self, config):
        replay_buffer_size = config['replay_buffer_size']
        seed = config['seed']
        nr.seed(seed)

        self.replay_buffer_size = replay_buffer_size
        self.obs = deque([], maxlen=self.replay_buffer_size)
        self.action = deque([], maxlen=self.replay_buffer_size)
        self.reward = deque([], maxlen=self.replay_buffer_size)
        self.next_obs = deque([], maxlen=self.replay_buffer_size)
        self.done = deque([], maxlen=self.replay_buffer_size)

    def append_memory(self,
                      obs,
                      action,
                      reward,
                      next_obs,
                      done: bool):
        self.obs.append(obs)
        self.action.append(action)
        self.reward.append(reward)
        self.next_obs.append(next_obs)
        self.done.append(done)

    def sample(self, batch_size):
        buffer_size = len(self.obs)

        idx = nr.choice(buffer_size,
                        size=min(buffer_size, batch_size),
                        replace=False)
        t = Transitions
        t.obs = torch.stack(list(map(self.obs.__getitem__, idx)))
        t.action = torch.stack(list(map(self.action.__getitem__, idx)))
        t.reward = torch.stack(list(map(self.reward.__getitem__, idx)))
        t.next_obs = torch.stack(list(map(self.next_obs.__getitem__, idx)))
        t.done = torch.tensor(list(map(self.done.__getitem__, idx)))[:, None]
        return t

    def clear(self):
        self.obs = deque([], maxlen=self.replay_buffer_size)
        self.action = deque([], maxlen=self.replay_buffer_size)
        self.reward = deque([], maxlen=self.replay_buffer_size)
        self.next_obs = deque([], maxlen=self.replay_buffer_size)
        self.done = deque([], maxlen=self.replay_buffer_size)
    

###########################################################
###    Training for the 'MountainCar' Environment     #####
###########################################################

Tensor = torch.DoubleTensor
torch.set_default_tensor_type(Tensor)

from gym.wrappers import Monitor
env = Monitor(gym.make('MountainCar-v0'), './videoMountainCar', force=True)

steps = 0  # total number of steps
for i_episode in range(1):
    observation = env.reset()
    done = False
    t = 0  # time steps within each episode
    ret = 0.  # episodic return
    while done is False:
        env.render()  # render to screen

        obs = torch.tensor(env.state)  # observe the environment state


        if ( obs[1] > 0):
            action = 2
        else:
            action = 0

        next_obs, reward, done, info = env.step(action)  # environment advance to next step

        t += 1
        steps += 1
        ret += reward  # update episodic return
        if done:
            print("Episode {} finished after {} timesteps".format(i_episode, t+1))

env.close()