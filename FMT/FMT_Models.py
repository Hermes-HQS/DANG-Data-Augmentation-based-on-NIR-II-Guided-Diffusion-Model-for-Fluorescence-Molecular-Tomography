from torchinfo import summary
import torch.nn as nn
import torch
from torch.nn import functional as F
from torch.nn import init
from torchvision import transforms as T
import numpy as np
import math

class LayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        """Construct a layernorm module in the TF style (epsilon inside the square root).
        """
        super(LayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias



class IPS(nn.Module):
    def __init__(self):
        super(IPS, self).__init__()
        input_node = 2723
        output_node = 1965
        self.IPS_model = nn.Sequential()
        # please refer to our previous paper for the model structure

    def forward(self, x):
        output = self.IPS_model(x)
        return output


class KNNLC(nn.Module):
    def __init__(self, device):
        super(KNNLC, self).__init__()
        input_node = 2723
        output_node = 1965
        self.KNNLC_model = nn.Sequential()
        # please refer to our previous paper for the model structure

    def forward(self, x):
        output = self.KNNLC_model(x)
        return output


class EFCN(nn.Module):
    def __init__(self):
        super(EFCN, self).__init__()
        input_node = 2723
        output_node = 1965
        self.EFCN_model = nn.Sequential()
        # please refer to our previous paper for the model structure

    def forward(self, x):
        output = self.EFCN_model(x)
        return output
        



if __name__ == "__main__":
    net_test = IPS()
    summary(net_test, input_size=(16, 2723))
