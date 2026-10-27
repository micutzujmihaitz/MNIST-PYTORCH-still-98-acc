import numpy as np
import torch  
import torch.nn as nn 

class Retea(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784,128)
        self.fc2 = nn.Linear(128,10)
        __init__ : nn.init.kaiming_normal_(self.fc1.weight,nonlinearity="relu")
        __init__ : nn.init.kaiming_normal_(self.fc1.bias) 
        __init__ : nn.init.kaiming_normal_(self.fc2.weight,nonlinearity="relu")
        __init__ : nn.init.kaiming_normal_(self.fc2.bias)    
    def forward(self,x):
        z1 = self.fc1(x)
        a1 = torch.relu(z1)
        return self.fc2(a1)