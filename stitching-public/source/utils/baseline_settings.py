import torch
import torch.nn as nn
import numpy as np

def sign_acc_fn(y_pred, y_true):
    return torch.mean(torch.eq(y_true, torch.max(y_pred, dim=1)[1]).type(torch.FloatTensor))

def binary_sign_acc_fn(y_pred, y_true):
    return torch.mean(((torch.sign(y_pred.squeeze_())+1)/2 == y_true.squeeze_()).type(torch.FloatTensor))

def binary_rep_sign_fn(y_pred, y_true):
    return torch.mean((torch.sum(((torch.sign(y_pred)+1)/2 == y_true), dim=1)/len(y_pred[0])).type(torch.FloatTensor))

def get_baseline_loss(settings):
    if settings.losstype=='mse_classification':
        loss = nn.MSELoss()
        return lambda target, prediction: loss(prediction, torch.eye(settings.numout, device=target.device)[target])
    elif settings.losstype=='mse_regression':
        loss = nn.MSELoss()
        return lambda target, prediction: loss(prediction, target)
    elif settings.losstype=='ce':
        loss = nn.CrossEntropyLoss()
        return lambda target, prediction: loss(prediction, target)
    elif settings.losstype=='bce':
        loss = nn.BCEWithLogitsLoss()
        return lambda target, prediction: loss(prediction, target)
    else:
        raise NotImplementedError

def get_baseline_error(settings):
    if settings.errortype=='label_int':
        return lambda prediction, target: sign_acc_fn(prediction, target)
    elif settings.errortype=='label_bnry':
        print('hey')
        return lambda prediction, target: binary_rep_sign_fn(prediction, target)
    else:
        raise NotImplementedError
    

def get_init_vals(settings, num_trainpoints, classifier, norm_function):
    (weight_norm, bias_norm, bn_weight_norm, bn_bias_norm) = classifier.get_weight_bias_norms(norm_function)
    num_weight_norm_terms = len(weight_norm)
    if settings.gpu:
        if settings.layerwise_norm_coeff is None:
            if settings.normtype=='L2':
                layerwise_norm_coeff = 0.5*torch.ones(num_weight_norm_terms).cuda()
            else:
                layerwise_norm_coeff = torch.ones(num_weight_norm_terms).cuda()
        else:
            assert(len(layerwise_norm_coeff)==num_weight_norm_terms)
            layerwise_norm_coeff = torch.Tensor(layerwise_norm_coeff).cuda()
    else:
        if settings.layerwise_norm_coeff is None:
            if settings.normtype=='L2':
                layerwise_norm_coeff = 0.5*torch.ones(num_weight_norm_terms)
            else:
                layerwise_norm_coeff = torch.ones(num_weight_norm_terms)
        else:
            assert(len(layerwise_norm_coeff)==num_weight_norm_terms)
            layerwise_norm_coeff = torch.Tensor(layerwise_norm_coeff)
    return layerwise_norm_coeff
