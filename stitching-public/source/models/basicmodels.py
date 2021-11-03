import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

class Flatten(nn.Module):
    def forward(self, x): return x.view(x.size(0), -1)

def create_cnn(resolution, d_output, c_in, c_first, n_layers, batch_norm=False):
    layers = []
    channels = [c_in] + [c_first * 2**i for i in range(n_layers)]
    for i, (c1, c2) in enumerate(zip(channels[:-1], channels[1:])):
        layers += [nn.Conv2d(c1, c2, 3, padding=1)]
        if batch_norm:
            layers += [nn.BatchNorm2d(c2)]
        layers += [nn.ReLU()]
        if i == n_layers - 1:
            layers += [nn.AvgPool2d(resolution)]
        elif i != 0:
            layers += [nn.MaxPool2d(2)]
            resolution //= 2
    layers += [Flatten(), nn.Linear(channels[-1], d_output)]
    return nn.Sequential(*layers)

def create_fc(in_size, depth, numhid, numout, nonlin_name=None, batch_norm=False, add_bias=True):
    if nonlin_name=='relu':
        nonlinearity = nn.ReLU()

    if depth>1:
        layers = [Flatten(), nn.Linear(in_size, numhid, bias=add_bias)]
        if batch_norm:
            layers += [nn.BatchNorm1d(numhid)]
        if nonlin_name is not None:
            layers += [nonlinearity]
    else:
        layers = [Flatten(), nn.Linear(in_size, numout, bias=add_bias)]
        
            
    for d in range(depth-2):
        layers += [nn.Linear(numhid, numhid, bias=add_bias)]
        if batch_norm:
            layers += [nn.BatchNorm1d(numhid)]
        if nonlin_name is not None:
            layers += [nonlinearity]
    if depth>1: layers += [nn.Linear(numhid, numout, bias=add_bias)]
    print(nn.Sequential(*layers))
    return nn.Sequential(*layers)
