import torch.nn as nn
import torch.nn.functional as F

# network define
class FullyConnectedNet(nn.Module):
    
    def __init__(self):
        super(FullyConnectedNet, self).__init__()
        self.l1 = nn.Linear(42, 256)
        self.l2 = nn.Linear(256, 128)
        self.l3 = nn.Linear(128, 3)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.l1(x)
        x = self.relu(x)
        x = self.l2(x)
        x = self.relu(x)
        x = self.l3(x)
        return F.log_softmax(x)