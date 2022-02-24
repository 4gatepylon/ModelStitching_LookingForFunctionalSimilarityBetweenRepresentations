from tqdm import tqdm
import torch.nn.functional as F
import torch
import random
import numpy as np
from torch.cuda.amp import autocast
import math

def fix_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# modified from
# https://colab.research.google.com/github/facebookresearch/moco/blob/colab-notebook/colab/moco_cifar10_demo.\
# ipynb#scrollTo=RI1Y8bSImD7N
# test using a knn monitor
def knn_monitor(net, memory_data_loader, test_data_loader, memory_dataset, device='cuda', k=200, t=0.1,
                hide_progress=False,
                targets=None):

    if not targets:
        targets = memory_dataset.targets
    net.eval()

    classes = len(memory_dataset.classes)
    total_top1, total_top5, total_num, feature_bank = 0.0, 0.0, 0, []

    with torch.no_grad():
        # generate feature bank
        for data, target in tqdm(memory_data_loader, desc='Feature extracting', leave=False, disable=hide_progress):
            feature = net(data.to(device=device, non_blocking=True))
            feature = F.normalize(feature, dim=1)
            feature_bank.append(feature)

        # [D, N]
        feature_bank = torch.cat(feature_bank, dim=0).t().contiguous()
        # [N]
        feature_labels = torch.tensor(targets, device=feature_bank.device)

        # loop test data to predict the label by weighted knn search
        test_bar = tqdm(test_data_loader, desc='kNN', disable=hide_progress)
        for data, target in test_bar:
            data, target = data.to(device=device, non_blocking=True), target.to(device=device, non_blocking=True)
            feature = net(data)
            feature = F.normalize(feature, dim=1)
            pred_labels = knn_predict(feature, feature_bank, feature_labels, classes, k, t)
            total_num += data.size(0)
            total_top1 += (pred_labels[:, 0] == target).float().sum().item()
            test_bar.set_postfix({'Accuracy': total_top1 / total_num * 100})

    return total_top1 / total_num * 100


# knn monitor as in InstDisc https://arxiv.org/abs/1805.01978
# implementation follows http://github.com/zhirongw/lemniscate.pytorch and https://github.com/leftthomas/SimCLR
def knn_predict(feature, feature_bank, feature_labels, classes, knn_k, knn_t):
    # compute cos similarity between each feature vector and feature bank ---> [B, N]
    sim_matrix = torch.mm(feature, feature_bank)

    # [B, K]
    sim_weight, sim_indices = sim_matrix.topk(k=knn_k, dim=-1)

    # [B, K]
    sim_labels = torch.gather(feature_labels.expand(feature.size(0), -1), dim=-1, index=sim_indices)
    sim_weight = (sim_weight / knn_t).exp()

    # counts for each class
    one_hot_label = torch.zeros(feature.size(0) * knn_k, classes, device=sim_labels.device)

    # [B*K, C]
    one_hot_label = one_hot_label.scatter(dim=-1, index=sim_labels.view(-1, 1), value=1.0)

    # weighted score ---> [B, C]
    pred_scores = torch.sum(one_hot_label.view(feature.size(0), -1, classes) * sim_weight.unsqueeze(dim=-1), dim=1)
    pred_labels = pred_scores.argsort(dim=-1, descending=True)
    return pred_labels

def evaluate(model, test_loader):
    for layer in model:
        layer.eval()

    for idx, (images, labels) in enumerate(test_loader):
        total_correct, total_num = 0., 0.

        with torch.no_grad():
            with autocast():
                h = images
                for layer in model:
                    h = layer(h)
                preds = h.argmax(dim=1)
                total_correct = (preds == labels).sum().cpu().item()
                total_num += h.shape[0]

    return total_correct / total_num * 100.

def adjust_learning_rate(epochs, warmup_epochs, base_lr, optimizer, loader, step):
    max_steps = epochs * len(loader)
    warmup_steps = warmup_epochs * len(loader)
    if step < warmup_steps:
        lr = base_lr * step / warmup_steps
    else:
        step -= warmup_steps
        max_steps -= warmup_steps
        q = 0.5 * (1 + math.cos(math.pi * step / max_steps))
        end_lr = 0
        lr = base_lr * q + end_lr * (1 - q)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return lr

