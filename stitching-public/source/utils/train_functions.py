from torch.autograd import Variable
import torch
import numpy as np
from misc import VarToNumpy, to_categorical

def evaluate(test_data_loader, gpu, classifier, criterion=None, sign_acc_fn=None, log_predictions=False):
    test_loss = 0
    test_sign_acc = 0
    total_test_loss = 0
    total_test_acc = 0
    num_testpoints = 0
    if log_predictions: total_test_prediction = torch.LongTensor(max(test_data_loader.sampler.indices)+1)

    classifier.eval()

    with torch.no_grad():
        for (i_test, X) in enumerate(test_data_loader):
            test_input = X[0]
            test_target = X[1]
            if gpu:
                test_input = test_input.cuda(non_blocking=True)
                test_target = test_target.cuda(non_blocking=True)
            test_prediction = classifier(test_input)
            if log_predictions: total_test_prediction[X[2]] = torch.max(test_prediction, dim=1)[1].cpu()

            num_testpoints += test_input.size(0)
            if criterion is not None:
                test_loss += test_input.size(0)*VarToNumpy(criterion(test_target, test_prediction))
            if sign_acc_fn is not None:
                test_sign_acc += test_input.size(0)*VarToNumpy(sign_acc_fn(test_prediction, test_target))

    if criterion is not None:
        total_test_loss = test_loss/num_testpoints
    else:
        total_test_loss = None
        
    if sign_acc_fn is not None:
        total_test_acc = test_sign_acc/num_testpoints
    else:
        total_test_acc = None

    if log_predictions:
        return (total_test_prediction, total_test_loss, total_test_acc)
    else:
        return (None, total_test_loss, total_test_acc)

def evaluate_binary(test_data_loader, gpu, classifier, criterion=None, sign_acc_fn=None, log_predictions=False):
    test_loss = 0
    test_sign_acc = 0
    total_test_loss = 0
    total_test_acc = 0
    num_testpoints = 0
    if log_predictions: total_test_prediction = np.empty(max(test_data_loader.sampler.indices)+1, dtype=np.uint16)

    classifier.eval()
    
    for (i_test, X) in enumerate(test_data_loader):
        test_input = Variable(X[0], volatile=True)
        test_target = Variable(X[1], volatile=True)
        if gpu:
            test_input = test_input.cuda()
            test_target = test_target.cuda()
        test_prediction = classifier(test_input)

        binary_prediction = ((torch.sign(test_prediction)+1)/2).cpu().detach().numpy().astype(np.uint8)
        index_prediction = np.packbits(binary_prediction.reshape(-1, 2, 8)[:, ::-1]).view(np.uint16)
        
        if log_predictions: total_test_prediction[X[2]] = index_prediction
            
        num_testpoints += test_input.size(0)
        if criterion is not None:
            test_loss += test_input.size(0)*VarToNumpy(criterion(test_target, test_prediction))
        if sign_acc_fn is not None:
            test_sign_acc += test_input.size(0)*VarToNumpy(sign_acc_fn(test_prediction, test_target))

    if criterion is not None:
        total_test_loss = test_loss/num_testpoints
    else:
        total_test_loss = None
        
    if sign_acc_fn is not None:
        total_test_acc = test_sign_acc/num_testpoints
    else:
        total_test_acc = None

    if log_predictions:
        return (total_test_prediction, total_test_loss, total_test_acc)
    else:
        return (None, total_test_loss, total_test_acc)    


def inclusion_mask(eval_inds, set_inds):
    return [ind in set_inds for ind in eval_inds]


def evaluate_noisy(test_data_loader, gpu, classifier, corrupt_inds, criterion=None, sign_acc_fn=None, log_predictions=False):
    test_loss = 0
    test_sign_acc = 0
    correct_prob = 0
    total_test_loss = 0
    total_test_acc = 0
    num_testpoints = 0
    if log_predictions: total_test_prediction = torch.LongTensor(max(test_data_loader.sampler.indices)+1)

    classifier.eval()
    
    for (i_test, X) in enumerate(test_data_loader):
        test_input = Variable(X[0], volatile=True)
        test_target = Variable(X[1], volatile=True)
        if gpu:
            test_input = test_input.cuda()
            test_target = test_target.cuda()
        test_prediction = classifier(test_input)
        if log_predictions: total_test_prediction[X[2]] = torch.max(test_prediction, dim=1)[1].cpu()
        noisy_indices = inclusion_mask(X[2].cpu().numpy(), corrupt_inds)
        test_target = test_target[np.where(noisy_indices)[0]]
        test_prediction = test_prediction[np.where(noisy_indices)[0]]
        num_testpoints += len(np.where(noisy_indices)[0])
        
        if criterion is not None and len(np.where(noisy_indices)[0])>0:
            test_loss += len(test_target)*VarToNumpy(criterion(test_target, test_prediction))
        if sign_acc_fn is not None and len(np.where(noisy_indices)[0])>0:
            test_sign_acc += len(test_target)*VarToNumpy(sign_acc_fn(test_prediction, test_target))

    if criterion is not None:
        total_test_loss = test_loss/num_testpoints
    else:
        total_test_loss = None
        
    if sign_acc_fn is not None:
        total_test_acc = test_sign_acc/num_testpoints
    else:
        total_test_acc = None

    if log_predictions:
        return (total_test_prediction, total_test_loss, total_test_acc)
    else:
        return (None, total_test_loss, total_test_acc)


def evaluate_confusion_matrix(dataset, predictions, No, Nfine):
    conf_mat = np.zeros((No, Nfine))
    for j in range(Nfine):
        if j in np.unique(dataset.fine_labels):
            j_indices = list(np.where(np.array(dataset.fine_labels)==j)[0])
            conf_mat[:, j] = np.histogram(predictions[j_indices], bins = np.arange(No+1))[0]

    return conf_mat/len(dataset.fine_labels)
