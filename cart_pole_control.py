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
    

import numpy as np

def solve_lqr(A, B, Q, R):
    """
    Solve the Linear Quadratic Regulator (LQR) problem.

    Args:
        A (numpy.ndarray): State transition matrix.
        B (numpy.ndarray): Control input matrix.
        Q (numpy.ndarray): State cost matrix.
        R (numpy.ndarray): Control cost matrix.

    Returns:
        numpy.ndarray: Optimal gain matrix K.
    """
    P = np.matrix(np.zeros_like(Q))
    K = np.matrix(np.zeros((B.shape[1], A.shape[0])))

    # Solve the Algebraic Riccati equation
    # P = A^T * P * A - A^T * P * B * (R + B^T * P * B)^-1 * B^T * P * A + Q
    max_iterations = 1000
    eps = 0.01
    for _ in range(max_iterations):
        P_new = A.T * P * A - A.T * P * B * np.linalg.inv(R + B.T * P * B) * B.T * P * A + Q
        if np.max(np.abs(P - P_new)) < eps:
            P = P_new
            break
        P = P_new

    # Calculate the optimal gain matrix K
    K = np.linalg.inv(R + B.T * P * B) * B.T * P * A

    return K

def cartpole_lqr():
    """
    Solves the cartpole control problem using LQR.

    Returns:
        numpy.array: LQR gain matrix.
    """
    # System dynamics
    m_c = 1.0  # Mass of the cart
    m_p = 0.1  # Mass of the pole
    l = 0.5    # Length of the pole
    g = 9.8    # Acceleration due to gravity

    A = np.array([[0, 1, 0, 0],
                  [0, 0, g/(l*(4.0/3 - m_p/(m_p+m_c))), 0],
                  [0, 0, 0, 1],
                  [0, 0, g/(l*(4.0/3 - m_p/(m_p+m_c))), 0]])

    B = np.array([[0],
                  [1 / (m_c + m_p)],
                  [0],
                  [-1/(l*(4.0/3 - m_p/(m_p+m_c)))]])

    # Cost matrices
    Q = np.diag([1, 1, 10, 1])  # State cost matrix
    R = np.array([[0.001]])    # Control cost matrix

    # Solve LQR
    P = linalg.solve_continuous_are(A, B, Q, R)
    K = np.dot(np.linalg.inv(R), np.dot(B.T, P))
    return K


# Solve the LQR problem
K = cartpole_lqr()
K = np.squeeze(K)
print(K)
print(np.shape(K))
K = torch.from_numpy(K)

###########################################################
###     Training for the 'Chart-Pole' Environment     #####
###########################################################

Tensor = torch.DoubleTensor
torch.set_default_tensor_type(Tensor)

from gym.wrappers import Monitor
env = Monitor(gym.make('CartPole-v0'), './videoCartPole', force=True)

steps = 0  # total number of steps 
reference_pos = torch.zeros((4,))
reference_pos[0] = 1
for i_episode in range(1):
    observation = env.reset()
    done = False
    t = 0  # time steps within each episode
    ret = 0.  # episodic return
    while done is False:
        env.render()  # render to screen

        obs = torch.tensor(env.state)  # observe the environment state
        error1 = reference_pos - obs
        action = torch.sign(torch.inner(K, error1))
        if action < 0:
            action = 0
        else:
            action = 1
        
        
        next_obs, reward, done, info = env.step(action)  # environment advance to next step

        t += 1
        steps += 1
        ret += reward  # update episodic return
        if done:
            print("Episode {} finished after {} timesteps".format(i_episode, t+1))

env.close()