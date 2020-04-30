import torch
import torch.nn as nn
import torch.nn.functional as F

class QNetwork(nn.Module):
    def __init__(self,state_size, action_size):
        super(QNetwork, self).__init__()
        
        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64,64)
        #self.fc3 = nn.Linear(64,64)
        self.fc4 = nn.Linear(64,action_size)
        
        
    def forward(self, x):
        
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        #x = F.relu(self.fc3(x))
        x = self.fc4(x)
        
        return x
    
  


 class DuelingDQN(nn.Module):
    def __init__(self,state_size, action_size):
        super(DuelingDQN, self).__init__()
        
        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64,64)
        
        
        self.value_stream = nn.Linear(64, 1)
        self.advantage_stream = nn.Linear(64, action_size)
        
        
    def forward(self, x):
        
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        
        
        values = self.value_stream(x)
        advantages = self.advantage_stream(x)
        
        out = values + (advantages - advantages.mean())
        
        return out
    
    
    
    
    
    
    
    
    