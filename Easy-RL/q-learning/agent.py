import numpy as np
import torch
import math
from collections import defaultdict

class QLearning(object):
    def __init__(self, state_dim, action_dim, config):
        self.state_dim = state_dim
        self.action_dim = action_dim
        