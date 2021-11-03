import torch, time, math
import numpy as np

def to_categorical(y, num_classes):
    """ 1-hot encodes a tensor """
    return torch.eye(num_classes)[y]
#return torch.eye(num_classes, device=y.device)[y]

def VarToNumpy(x):
    if x.is_cuda:
        return x.data.cpu().numpy()
    else:
        return x.data.numpy()

def timeSince(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

def sign_acc_fn(y_pred, y_true):
    return torch.mean(torch.eq(y_true, torch.max(y_pred, dim=1)[1]).type(torch.FloatTensor))

def binary_sign_acc_fn(y_pred, y_true):
    return torch.mean(((torch.sign(y_pred.squeeze_())+1)/2 == y_true.squeeze_()).type(torch.FloatTensor))

def binary_rep_sign_fn(y_pred, y_true):
    return torch.sum(((torch.sign(y_pred)+1)/2 == y_true), dim=1)/len(y_pred[0])

def const_val_epoch(total_train_acc, const_val, const_epochs):
    '''
    First epoch at which training accuracy is constant at a given value for at least constant_epoch values
    '''
    count = np.zeros([len(total_train_acc)]).astype(int)
    for i in range(len(total_train_acc)):
        if total_train_acc[i]==1.:
            if i>0:
                count[i]+=count[i-1]+1
            else:
                count[i]=1
        else:
            count[i]=0
        if count[i]==const_epochs:
            return (i-const_epochs+1)
        else:
            return 0

def accuracy(output, target, topk=(1,)):
    pred = output.topk(max(topk), 1, True, True)[1].t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    return [correct[:k].reshape(-1).float().sum(0, keepdim=True).cpu() for k in topk]

def clip_acc_from_weights(image_features, target, zeroshot_weights):
    image_features /= image_features.norm(dim=-1, keepdim=True)
    logits = 100. * image_features @ zeroshot_weights

    # measure accuracy
    acc1 = sign_acc_fn(logits, target)

    return acc1

def clip_im_zero_shot_acc(zero_shot_weights_path):
    zero_shot_weights = torch.load(zero_shot_weights_path).float().cuda()
    return lambda feat, targets: clip_acc_from_weights(feat, targets, zero_shot_weights)
