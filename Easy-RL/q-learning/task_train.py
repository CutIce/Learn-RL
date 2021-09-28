import torch
import os, sys, math

import numpy as np
from gym.envs.toy_text import discrete
from collections import defaultdict


class QLearning(object):
    def __init__(self, state_dim, action_dim, config):
        super(QLearning, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.lr = config['lr']
        self.gamma = config['gamma']
        self.episilon = 0
        self.episilon_start = config['episilon_start']
        self.episilon_end = config['episilon_end']
        self.sample_count = 0
        self.episilon_decay = config['episilin_decay']
        self.Q_table = defaultdict(lambda: np.zeros(action_dim))


    def choose_action(self, state):
        self.sample_count += 1
    
        self.episilon = self.episilon_end + (self.episilon_start - self.episilon_end) * math.exp(-1.0 * self.sample_count / self.episilon_decay)

        if np.random.uniform(0, 1) > self.epsilon:
            action = np.argmax(self.Q_table[str(state)])
        else:
            action = np.random.choice(self.action_dim)

        return action


    def predict(self, state):
        action = np.argmax(self.Q_table[str(state)])
        return action

    
    def update(self, state, action, reward, next_state, done):
        predict = self.Q_table[str(state)][action]
        if done:
            target = reward
        else:
            target = reward + self.gamma * np.max(self.Q_table[str(next_state)])
        self.Q_table[str(state)][action] += self.lr * (target - predict)

    
    def save(self, path='./checkpoint/QLearning_model.pkl'):
        import dill
        if not os.path.exists(path):
            os.makedirs(path)
        torch.save(obj=self.Q_table, f=path, pickle_module=dill)
        print("Saving successfully !")

    def load(self, path='./checkpoint/QLearning_model.pkl'):
        assert (not os.path.exists(path)), f"No Model in {path}"
        import dill
        self.Q_table = torch.load(path, pickle_module=dill)
        print("Loading Successfully !")