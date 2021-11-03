import torch
import torchvision.transforms as transforms
import torchvision.datasets as dsets
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
import numpy.random as npr

import numpy as np
from PIL import Image
import os
import os.path
import errno
import codecs
import hashlib
import sys
if sys.version_info[0] == 2:
    import cPickle as pickle
else:
    import pickle
import pickle as pkl

import copy

def get_transform_cub():
    scale = 256.0/224.0
    target_resolution = (224, 224)

    # Resizes the image to a slightly larger square then crops the center.
    transform = transforms.Compose([
        transforms.Resize((int(target_resolution[0]*scale), int(target_resolution[1]*scale))),
        transforms.CenterCrop(target_resolution),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    return transform

def get_transform(dataname, classifiername, data_augment, transform_name, dlaas, settings):
    '''
    Returns pair transform_train, transform_test for the given dataset, classifier and the option to data augment
    '''

    if dlaas:
        sys.path.insert(0, 'source/utils')
    else:
        sys.path.insert(0, '../source/utils')

    from autoaugment import Cutout, CIFAR10Policy
    from corruptions import GaussianNoise, GaussianNoiseSelect
    
    if dataname == 'ToyDataset' or dataname == 'ToyDatasetRegression':
        return None, None
    elif dataname == 'CIFAR10' or dataname == 'CIFAR10withInds' or dataname == 'CIFAR10withIndsCorrupt':
        if transform_name == 'common':
            mean = np.array([0.4914, 0.4822, 0.4465])
            std = np.array([0.2470, 0.2435, 0.2616])
            if data_augment:
                transform_train = transforms.Compose([
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(mean, std),
                ])
            else:
                transform_train = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize(mean, std),
                ])
            transform_test = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ])
        elif transform_name == 'simclr':
            transform_train = transforms.Compose([
                transforms.RandomResizedCrop(32),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
                transforms.RandomGrayscale(p=0.2),
                transforms.ToTensor(),
                transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
            ])

            transform_test = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
            ])
            
        elif transform_name == 'corruption':
            mean = np.array([0.4914, 0.4822, 0.4465])
            std = np.array([0.2470, 0.2435, 0.2616])
            transform_train = transforms.Compose(
                        [transforms.ToTensor(),
                         GaussianNoiseSelect(settings.gnoise_std)
                         ])
            transform_test = transforms.Compose([
                transforms.ToTensor()
            ])                        
            
        elif transform_name == 'autoaugment':
            mean = np.array([0.5071, 0.4867, 0.4408])
            std = np.array([0.2675, 0.2565, 0.2761])
            transform_train = transforms.Compose(
                        [transforms.RandomCrop(32, padding=4), # fill parameter needs torchvision installed from source
                         transforms.RandomHorizontalFlip(), CIFAR10Policy(), 
			 transforms.ToTensor(), 
                         Cutout(n_holes=1, length=16), # (https://github.com/uoguelph-mlrg/Cutout/blob/master/util/cutout.py)
                         transforms.Normalize(mean, std)])
            transform_test = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ])                        
        else:
            transform_train = transforms.Compose([
                transforms.ToTensor(),
            ])

            transform_test = transforms.Compose([
                transforms.ToTensor(),
            ])
    elif dataname == 'CIFAR100' or dataname == 'CIFAR100withInds' or dataname == 'CIFAR100withIndsCorrupt':
        
        if transform_name == 'common':
            mean = np.array([0.5071, 0.4867, 0.4408])
            std = np.array([0.2675, 0.2565, 0.2761])
            if data_augment:
                transform_train = transforms.Compose([
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(mean, std),
                ])
            else:
                transform_train = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize(mean, std),
                ])
            transform_test = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ])
        elif transform_name == 'simclr':
            mean = np.array([0.4914, 0.4822, 0.4465])
            std = np.array([0.2023, 0.1994, 0.2010])
            if data_augment:
                transform_train = transforms.Compose([
                    transforms.RandomResizedCrop(32),
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
                    transforms.RandomGrayscale(p=0.2),
                    transforms.ToTensor(),
                    transforms.Normalize(mean, std)
                ])
            else:
                transform_train = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize(mean, std),
                ])
            transform_test = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ])                
        elif transform_name == 'autoaugment':
            mean = np.array([0.5071, 0.4867, 0.4408])
            std = np.array([0.2675, 0.2565, 0.2761])
            transform_train = transforms.Compose(
                        [transforms.RandomCrop(32, padding=4), # fill parameter needs torchvision installed from source
                         transforms.RandomHorizontalFlip(), CIFAR10Policy(), 
			 transforms.ToTensor(), 
                         Cutout(n_holes=1, length=16), # (https://github.com/uoguelph-mlrg/Cutout/blob/master/util/cutout.py)
                         transforms.Normalize(mean, std)])
            transform_test = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ])            
        else:
            transform_train = transforms.Compose([
                transforms.ToTensor(),
            ])

            transform_test = transforms.Compose([
                transforms.ToTensor(),
            ])
    elif dataname == 'ImageNet':
        mean=[0.485, 0.456, 0.406]
        std=[0.229, 0.224, 0.225]
        if data_augment and transform_name == 'common':
            transform_train = transforms.Compose([
                                 transforms.RandomResizedCrop(224),
                                 transforms.RandomHorizontalFlip(),
                                 transforms.ToTensor(),
                                 transforms.Normalize(mean, std)
                             ])
            transform_test=transforms.Compose([
                                 transforms.Resize(256),
                                 transforms.CenterCrop(224),
                                 transforms.ToTensor(),
                                 transforms.Normalize(mean,std)
                             ])
        else:
            transform_train = transforms.Compose([
                                 transforms.Resize(256),
                                 transforms.CenterCrop(256),
                                 transforms.ToTensor(),
                                 transforms.Normalize(mean,std)
                             ]) 
            transform_test=transforms.Compose([
                                 transforms.Resize(256),
                                 transforms.CenterCrop(256),
                                 transforms.ToTensor(),
                                 transforms.Normalize(mean,std)
                             ])
    elif dataname == 'ImageNet256':
        mean=[0.485, 0.456, 0.406]
        std=[0.229, 0.224, 0.225]
        if data_augment and transform_name == 'common':
            transform_train = transforms.Compose([
                                 transforms.RandomResizedCrop(224),
                                 transforms.RandomHorizontalFlip(),
                                 transforms.ToTensor(),
                                 transforms.Normalize(mean, std)
                             ])
            transform_test=transforms.Compose([
                                 transforms.CenterCrop(224),
                                 transforms.ToTensor(),
                                 transforms.Normalize(mean,std)
                             ])
        else:
            transform_train = transforms.Compose([
                                 transforms.CenterCrop(256),
                                 transforms.ToTensor(),
                                 transforms.Normalize(mean,std)
                             ]) 
            transform_test=transforms.Compose([
                                 transforms.CenterCrop(256),
                                 transforms.ToTensor(),
                                 transforms.Normalize(mean,std)
                             ])             
                                             

        
    elif dataname == 'CUBDataset':
        scale = 256.0/224.0
        target_resolution = (224, 224)

        # Resizes the image to a slightly larger square then crops the center.
        transform_train = transform_test = transforms.Compose([
            transforms.Resize((int(target_resolution[0]*scale), int(target_resolution[1]*scale))),
            transforms.CenterCrop(target_resolution),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    elif 'CelebA' in dataname:
        target_resolution = (224, 224)

        # Resizes the image to a slightly larger square then crops the center.
        transform_train = transform_test = transforms.Compose([
            transforms.Resize((int(target_resolution[0]), int(target_resolution[1]))),
            transforms.CenterCrop(target_resolution),
            transforms.ToTensor(),
        ])
        
    elif dataname == 'MNIST' or dataname == 'MNISTwithInds' or dataname == 'MNISTwithIndsCorrupt':
        transform_train = transforms.Compose([
                transforms.ToTensor(),
            ])

        transform_test = transforms.Compose([
                transforms.ToTensor(),
            ])
    return transform_train, transform_test


def get_dataset(dataname, data_dir, dlaas, classifiername, data_augment, transform_name, corruption_ratio=None, settings=None):
    
    if dlaas:
        sys.path.insert(0, 'source/utils')
    else:
        sys.path.insert(0, '../source/utils')

    from data import ToyToTorch, ToyToTorchRegression, CIFAR10withInds, MNISTwithInds, CIFAR10withIndsCorrupt, MNISTwithIndsCorrupt, CIFAR100withInds, CIFAR100withIndsCorrupt, CUBDataset
    from celeba import CelebA
    from autoaugment import CIFAR10Policy

    transform_train, transform_test = get_transform(dataname, classifiername, data_augment, transform_name, dlaas, settings)
    
    if dataname == 'ToyDataset':
        train_dataset = ToyToTorch(data_dir, train=True)
        test_dataset = ToyToTorch(data_dir, train=False)
    elif dataname == 'ToyDatasetRegression':
        train_dataset = ToyToTorchRegression(data_dir, train=True)
        test_dataset = ToyToTorchRegression(data_dir, train=False)
    elif dataname == 'CUBDataset':
        train_dataset = CUBDataset(root_dir=data_dir, train=True, transform=transform_train)
        test_dataset = CUBDataset(root_dir=data_dir, train=False, transform=transform_test)
    elif 'CelebA' in dataname:
        train_attr = dataname.split('-')[1]
        fine_attr = dataname.split('-')[2]
        train_dataset = CelebA(root=data_dir, train_attr=train_attr, fine_attr=fine_attr, split='train', transform=transform_train)
        test_dataset = CelebA(root=data_dir, train_attr=train_attr, fine_attr=fine_attr, split='test', transform=transform_test)
    elif dataname == 'ImageNet' or dataname == 'ImageNet256':
        train_dataset = dsets.ImageFolder(os.path.join(data_dir, 'train'), transform=transform_train)
        test_dataset = dsets.ImageFolder(os.path.join(data_dir, 'val'), transform=transform_test)
    elif dataname == 'CIFAR10':
        train_dataset = dsets.CIFAR10(root=data_dir,
                                        train=True,
                                        download=True,
                                        transform=transform_train)
        
        test_dataset = dsets.CIFAR10(root=data_dir,
                                       train=False,
                                       download=True,
                                       transform=transform_test)

    elif dataname == 'CIFAR10withInds':
        train_dataset = CIFAR10withInds(root=data_dir,
                                        train=True,
                                        download=True,
                                        transform=transform_train)
        
        test_dataset = CIFAR10withInds(root=data_dir,
                                       train=False,
                                       download=True,
                                       transform=transform_test)

    elif dataname == 'CIFAR10withIndsCorrupt':
        train_dataset = CIFAR10withIndsCorrupt(root=data_dir,
                                               corruption_ratio=corruption_ratio,
                                        train=True,
                                        download=True,
                                        transform=transform_train)
        
        test_dataset = CIFAR10withIndsCorrupt(root=data_dir,
                                              corruption_ratio=corruption_ratio,
                                       train=False,
                                       download=True,
                                       transform=transform_test)
    elif dataname == 'CIFAR100':
        train_dataset = dsets.CIFAR100(root=data_dir,
                                        train=True,
                                        download=True,
                                        transform=transform_train)
        
        test_dataset = dsets.CIFAR100(root=data_dir,
                                       train=False,
                                       download=True,
                                       transform=transform_test)
    elif dataname == 'CIFAR100withInds':
        train_dataset = CIFAR100withInds(root=data_dir,
                                        train=True,
                                        download=True,
                                        transform=transform_train)
        
        test_dataset = CIFAR100withInds(root=data_dir,
                                       train=False,
                                       download=True,
                                       transform=transform_test)
    elif dataname == 'CIFAR100withIndsCorrupt':
        train_dataset = CIFAR100withIndsCorrupt(root=data_dir,
                                        corruption_ratio=corruption_ratio,
                                        train=True,
                                        download=True,
                                        transform=transform_train)
        
        test_dataset = CIFAR100withIndsCorrupt(root=data_dir,
                                               corruption_ratio=corruption_ratio,
                                       train=False,
                                       download=True,
                                       transform=transform_test)
    elif dataname == 'MNIST':
        train_dataset = dsets.MNIST(root=data_dir,
                                        train=True,
                                        download=True,
                                        transform=transform_train)

        test_dataset = dsets.MNIST(root=data_dir,
                                       train=False,
                                       download=True,
                                       transform=transform_test)
    elif dataname == 'MNISTwithInds':
        train_dataset = MNISTwithInds(root=data_dir,
                                        train=True,
                                        download=True,
                                        transform=transform_train)

        test_dataset = MNISTwithInds(root=data_dir,
                                       train=False,
                                       download=True,
                                       transform=transform_test)
    elif dataname == 'MNISTwithIndsCorrupt':
        train_dataset = MNISTwithIndsCorrupt(root=data_dir,
                                             corruption_ratio=corruption_ratio,
                                        train=True,
                                        download=True,
                                        transform=transform_train)

        test_dataset = MNISTwithIndsCorrupt(root=data_dir,
                                            corruption_ratio=corruption_ratio,
                                       train=False,
                                       download=True,
                                       transform=transform_test)
    else:
        raise NotImplementedError

    return train_dataset, test_dataset

def make_dataloaders(numsamples, No, batchsize, testbatchsize, train_dataset, test_dataset, validation, shuffle=False, tr_workers=0, val_workers=0, test_workers=0, tr_val_workers=0, train_inds_root=None):
    '''
    Condition: max(train_inds_given) < numsamples
    '''

    total_trainpoints = len(train_dataset)
    total_train_inds = np.arange(0, total_trainpoints)
    if shuffle: total_train_inds = npr.permutation(total_train_inds)

    total_testpoints = len(test_dataset)
    test_inds = np.arange(0, total_testpoints)

    if train_inds_root is None:
        if numsamples == -1:
            num_trainpoints = total_trainpoints
        else:
            num_trainpoints = numsamples
        train_inds = total_train_inds[:num_trainpoints]
    else:
        train_inds_given = np.load(train_inds_root)
        assert max(train_inds_given)<numsamples, 'max(train_inds_given)<numsamples for correct validation'
        train_inds = train_inds_given
        num_trainpoints = len(train_inds)

    trainbsz = num_trainpoints*(batchsize==-1) + batchsize*(batchsize!=-1)
    testbsz = testbatchsize

    tr_data_loader = torch.utils.data.DataLoader(train_dataset, batch_size=trainbsz, sampler=torch.utils.data.sampler.SubsetRandomSampler(train_inds), num_workers=tr_workers)

    train_dataset_for_val = copy.deepcopy(train_dataset)
    train_dataset_for_val.transform = test_dataset.transform

    tr_data_loader_for_val = torch.utils.data.DataLoader(train_dataset_for_val, batch_size=testbsz, sampler=torch.utils.data.sampler.SubsetRandomSampler(train_inds), num_workers=tr_val_workers)

    test_data_loader = torch.utils.data.DataLoader(test_dataset, batch_size=testbsz, sampler=torch.utils.data.sampler.SubsetRandomSampler(test_inds), num_workers=test_workers)    

    ################# Validation #####################
    if validation:
        assert numsamples < total_trainpoints, 'Number of training samples should be strictly less than the total number of training points available in the validation phase'
        
        num_valpoints = total_trainpoints - numsamples
        val_inds = total_train_inds[num_trainpoints:]
        val_data_loader = torch.utils.data.DataLoader(train_dataset, batch_size=testbsz, sampler=torch.utils.data.sampler.SubsetRandomSampler(val_inds), num_workers=val_workers)

    else:
        val_data_loader = test_data_loader
        num_valpoints = total_testpoints

    return (tr_data_loader, tr_data_loader_for_val, val_data_loader, test_data_loader, num_trainpoints, trainbsz, num_valpoints, testbsz)
