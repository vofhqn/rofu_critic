import random
import numpy as np
from numpy.random import choice

# class ReplayMemory:
#     def __init__(self, capacity, seed):
#         random.seed(seed)
#         self.capacity = capacity
#         self.buffer = []
#         self.position = 0
#         self.weights = np.ones(capacity)

#     def push(self, state, action, qvalue):
#         if len(self.buffer) < self.capacity:
#             self.buffer.append(None)
#         self.buffer[self.position] = (state, action, qvalue)
#         self.position = (self.position + 1) % self.capacity

#     def sample(self, batch_size):
#         if batch_size>len(self.buffer):
#             batch_size=int(len(self.buffer)/2)
#         #batch = random.sample(self.buffer, batch_size)
#         sum_w = np.sum(self.weights[:len(self.buffer)])
#         batch = list(choice(self.buffer, batch_size, p=self.weights[:len(self.buffer)] / sum_w))
#         state, action, qvalue = map(np.stack, zip(*batch))
#         return state, action, qvalue

#     def __len__(self):
#         return len(self.buffer)

class ReplayMemory:
    def __init__(self, capacity, seed, state_dim, action_dim):
        random.seed(seed)
        self.capacity = capacity
        self.position = 0
        self.num_samples = 0
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.states = np.zeros((capacity, state_dim))
        self.actions = np.zeros((capacity, action_dim))
        self.qvalues = np.zeros(capacity)
        self.weights = np.ones(capacity)

    def push(self, state, action, qvalue, weight=1.):
        if self.num_samples < self.capacity:
            self.num_samples += 1

        self.states[self.position] = state
        self.actions[self.position] = action 
        self.qvalues[self.position] = qvalue
        self.weights[self.position] = weight
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        sum_w = np.sum(self.weights[:self.num_samples])
        batch = list(choice(self.num_samples, batch_size, p=self.weights[:self.num_samples] / sum_w))
        states = self.states[batch]
        actions = self.actions[batch]
        qvalues = self.qvalues[batch]
        return states, actions, qvalues
    def __len__(self):
        return self.num_samples

class TrajReplayMemory:
    def __init__(self, capacity, seed, state_dim, action_dim):
        random.seed(seed)
        self.capacity = capacity
        self.position = 0
        self.num_samples = 0
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.states = np.zeros((capacity, state_dim))
        self.actions = np.zeros((capacity, action_dim))
        self.rewards = np.zeros(capacity)
        self.nextstates = np.zeros((capacity, state_dim))
        self.masks = np.zeros(capacity)
        self.weights = np.ones(capacity)

    def push(self, state, action, reward, nextstate, mask, weight=1.):
        if self.num_samples < self.capacity:
            self.num_samples += 1

        self.states[self.position] = state
        self.actions[self.position] = action 
        self.rewards[self.position] = reward
        self.nextstates[self.position] = nextstate
        self.masks[self.position] = mask
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        sum_w = np.sum(self.weights[:self.num_samples])
        batch = list(choice(self.num_samples, batch_size, p=self.weights[:self.num_samples] / sum_w))
        states = self.states[batch]
        actions = self.actions[batch]
        rewards = self.rewards[batch]
        nextstates = self.nextstates[batch]
        masks = self.masks[batch]
        return states, actions, rewards, nextstates, masks
    def __len__(self):
        return self.num_samples