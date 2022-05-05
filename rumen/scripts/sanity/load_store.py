import os
import time
import random
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.cuda.amp import GradScaler, autocast


def fix_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# class MockDataset(Dataset):
#     """ Mocks dataset so that we can test other things (like loaders) """
#     NUM_SAMPLES = 10
#     Xs = [torch.rand((3, 32, 32)) for _ in range(NUM_SAMPLES)]
#     Ys = [torch.tensor(0) for _ in range(NUM_SAMPLES)]

#     def __init__(self):
#         self.X_Y = list(zip(MockDataset.Xs, MockDataset.Ys))

#     def __len__(self):
#         return len(self.X_Y)

#     def __getitem__(self, idx):
#         return self.X_Y[idx]


def pclone(model):
    return [p.data.detach().clone() for p in model.parameters()]


def listeq(l1, l2):
    return min((torch.eq(a, b).int().min().item() for a, b in zip(l1, l2))) == 1


# Simple script to check that if you load the same wieghts the model should be the same
if __name__ == "__main__":
    fix_seed(0)

    model = nn.Linear(10, 10)

    device = \
        torch.device("cuda") if torch.cuda.is_available(
        ) else torch.device("cpu")
    print("On device:", device)
    model = model.to(device)
    model.eval()
    torch.save(model.state_dict(), "model.pt")

    orig_weights = pclone(model)

    model2 = nn.Linear(10, 10)
    model2.load_state_dict(torch.load("model.pt"))
    model2 = model2.to(device)
    model2.eval()

    new_weights = pclone(model2)

    assert listeq(
        orig_weights, new_weights), "Weights should be equal on load/store"
