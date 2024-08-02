import torch.nn as nn
import torch


class GELU(nn.Module):
    
    def __init__(self):
        super().__init__()

    def forward(self,x):
        return 0.5 * x * (1 + torch.tanh(
            torch.sqrt(torch.tensor(2.0 / torch.pi)) * 
            (x + 0.44715 * torch.pow(x, 3))
        ))