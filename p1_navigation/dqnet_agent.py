import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from collections import deque, namedtuple
import random
from model import QNetwork

BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 64         # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR = 5e-4               # learning rate 
UPDATE_EVERY = 4        # how often to update the network

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class ReplayBuffer():
    def __init__(self, action_size, buffer_size, batch_size, seed):
        
        self.action_size = action_size
        self.memory = deque(maxlen = buffer_size)
        self.experience = namedtuple('Experience', field_names = ['state', 'action', 'reward','next_state', 'done'])
        self.seed = random.seed(seed)
        self.batch_size = batch_size
        
    
    def __len__(self):
        return len(self.memory)
    def add(self, state, action, reward, next_state, done):
        
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)
        
    def sample(self):
        
        experiences = random.sample(self.memory, k=self.batch_size)
        
        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)
        
        
        return (states, actions, rewards, next_states, dones)
    
#########Agent############################################

class Agent():
    def __init__(self,state_size, action_size,updated_type='dqn' ,seed=0):
        
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)
        self.updated_type = updated_type
        
        self.qnetwork_local = QNetwork(state_size, action_size).to(device)
        self.qnetwork_target = QNetwork(state_size, action_size).to(device)
        
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, seed)
        
        self.t_step = 0
        self.optimizer = torch.optim.Adam(self.qnetwork_local.parameters(), lr = LR)
        
    def step(self, state, action, reward, next_state, done):
        
        self.memory.add(state, action, reward, next_state, done)
        
        self.t_step = (self.t_step + 1) %UPDATE_EVERY
        
        if self.t_step ==0:
            
            if len(self.memory)>BATCH_SIZE:
                experiances = self.memory.sample()
                self.learn(experiances, GAMMA)
        
    
    def act(self, state, eps=0.):
        
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        
        self.qnetwork_local.eval()
        
        with torch.no_grad():
            action_state = self.qnetwork_local(state)
            
        self.qnetwork_local.train()
        
        if random.random()> eps:
            return np.argmax(action_state.cpu().data.numpy())
        else:
            return np.random.choice(np.arange(self.action_size))
        
        
    def learn(self, experiances, gamma):
        
        states, actions, rewards, next_states, dones = experiances
        
        
        if self.updated_type == 'dqn':
            qtarget_nexts = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1) # max(1)[0] return max values
            
        elif self.updated_type == 'double_dqn':
            local_actions = self.qnetwork_local(next_states).max(1)[1].unsqueeze(1) # max(1)[1] return max valu indices
            qtarget_nexts = self.qnetwork_target(next_states).gather(1, local_actions)
            
            
        
        qtargets = rewards + (gamma*qtarget_nexts)*(1-dones)
        
        qexpecteds = self.qnetwork_local(states).gather(1, actions)
        
        loss = F.mse_loss(qtargets, qexpecteds)
        
        self.optimizer.zero_grad()
        
        loss.backward()
        
        self.optimizer.step()
        
        self.soft_update(self.qnetwork_local, self.qnetwork_target, TAU)
        
        
        
    def soft_update(self, local_model, target_model, tau):
        
        for target_param , local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau* local_param.data + (1-tau)* target_param.data)
        
     

