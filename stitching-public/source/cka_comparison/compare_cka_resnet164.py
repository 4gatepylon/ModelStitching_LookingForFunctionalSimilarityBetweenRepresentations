import sys, os
sys.path.insert(0, '../source/models/')
from basicmodels import create_cnn
from resnet import resnet18k_cifar
from resnet_pytorch_image_classification import resnet_pic
import numpy as np
import torch
import wandb

import torchvision.transforms as transforms
import torchvision.datasets as dsets
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, '../source/utils')
from cka import gram_linear, cka, feature_space_linear_cka
from model_stitching import convert_model_to_Seq

from baseline_settings import get_baseline_loss
from optims_lr import get_optimizers, get_lr_scheduler
from classifier_settings import get_classifier
from misc import VarToNumpy, sign_acc_fn, timeSince, const_val_epoch
from train_functions import evaluate, evaluate_noisy
from data import ToyToTorch, ToyToTorchRegression, CIFAR10withInds, MNISTwithInds, CIFAR10withIndsCorrupt, MNISTwithIndsCorrupt, CIFAR100withInds, CIFAR100withIndsCorrupt, CUBDataset

with open('/n/coxfs01/ybansal/settings/wandb_api_key.txt', 'r') as f:
    key = f.read().split('\n')[0]
os.environ['WANDB_API_KEY'] = key
wandb.init(project='cka-compute', entity='yaminibansal')

layer_index = int(sys.argv[1])
layer_index2 = int(sys.argv[2])
model_name = 'resnet20'
path1 = '/n/coxfs01/ybansal/main-exp/rep-compare-models/3k3tfylg/final_model.pt'
path2 = '/n/coxfs01/ybansal/main-exp/rep-compare-models/3k3tfylg/final_model.pt'


wandb.run.summary[f"layer_ind"] = layer_index
wandb.run.summary[f"layer_ind2"] = layer_index2
wandb.run.summary[f"path1"] = path1
wandb.run.summary[f"path2"] = path2

model1 = torch.load(path1)
model2 = torch.load(path2)

new_model1 = convert_model_to_Seq(model1, model_name)
new_model2 = convert_model_to_Seq(model2, model_name)
print('Loaded models')

def feature_ex(classifier, X, classifiername, layer_ind=0):
    if classifiername=='mCNN_k_bn_cifar10':
        features = classifier[:layer_ind+1](X).view(X.shape[0], -1)
    elif 'resnet' in classifiername:
        features = classifier[:layer_ind+1](X).view(X.shape[0], -1)
    elif classifiername=='wideresnet_pic_myinit':
        if layer_ind==0:
            features = classifier.conv(X).view(X.shape[0], -1)
    else:
        raise NotImplementedError
    features = features.detach().cpu().numpy()
    return features

a = feature_ex(new_model1, torch.randn(4, 3, 32, 32).cuda(), model_name, layer_ind=layer_index)
b = feature_ex(new_model1, torch.randn(4, 3, 32, 32).cuda(), model_name, layer_ind=layer_index2)

mean = np.array([0.4914, 0.4822, 0.4465])
std = np.array([0.2470, 0.2435, 0.2616])

transform_train = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean, std),
])

data_dir = '/n/coxfs01/ybansal/Datasets/CIFAR10'
batch_size = 512
feature_size = a.shape[-1]
feature_size2 = b.shape[-1]
wandb.run.summary["feature_size"] = feature_size
wandb.run.summary["feature_size2"] = feature_size2

train_data = CIFAR10withInds(root=data_dir,
                                train=True,
                                download=True,
                                transform=transform_train)

test_data = CIFAR10withInds(root=data_dir,
                               train=False,
                               download=True,
                               transform=transform_test)
print('Loaded dataset')


num_classes = 10 #len(train_data.classes)
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

feature_ex_1 = lambda inp: feature_ex(new_model1, inp, model_name, layer_ind=layer_index)
feature_ex_2 = lambda inp: feature_ex(new_model2, inp, model_name, layer_ind=layer_index2)

test_size = len(test_data)

test_features_1 = np.zeros((test_size, feature_size), dtype='float32')
test_features_2 = np.zeros((test_size, feature_size2), dtype='float32')      

for (i_test, X) in enumerate(test_loader):
    print(i_test)
    inp = X[0].cuda(non_blocking=True)        
    target = X[1].cuda(non_blocking=True)
    inds = X[2]
    test_features_1[inds] = feature_ex_1(inp)

for (i_test, X) in enumerate(test_loader):
    print(i_test)
    inp = X[0].cuda(non_blocking=True)        
    target = X[1].cuda(non_blocking=True)
    inds = X[2]
    test_features_2[inds] = feature_ex_2(inp)
print('Computed features')

print('Computing CKA....')
cka_layer = cka(gram_linear(test_features_1), gram_linear(test_features_2))
print(cka_layer)
wandb.run.summary["cka"] = cka_layer
