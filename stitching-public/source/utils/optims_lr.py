import torch
from torch.optim import Optimizer
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from torch.optim.lr_scheduler import MultiStepLR
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR

required = object()

class OMD(Optimizer):
    r"""Implements optimistic mirror descent.

    OMD is based on the formula from
    `TRAINING GANS WITH OPTIMISM`__.

    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float): learning rate
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)

    Example:
        >>> optimizer = torch.optim.OMD(model.parameters(), lr=0.1)
        >>> optimizer.zero_grad()
        >>> loss_fn(model(input), target).backward()
        >>> optimizer.step()

    """

    def __init__(self, params, lr=required, momentum=0, dampening=0,
                 weight_decay=0, nesterov=False):
        defaults = dict(lr=lr, weight_decay=weight_decay)
        super(OMD, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(OMD, self).__setstate__(state)

    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            weight_decay = group['weight_decay']

            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data
                if weight_decay != 0:
                    d_p.add_(weight_decay, p.data)
                    
                param_state = self.state[p]
                if 'prev_grad_buffer' not in param_state:
                    buf = param_state['prev_grad_buffer'] = torch.zeros_like(p.data)
                else:
                    buf = param_state['prev_grad_buffer']

                p.data.add_(-2*group['lr'], d_p)
                p.data.add_(group['lr'], buf)

                param_state['prev_grad_buffer'] = torch.zeros_like(p.data)
                param_state['prev_grad_buffer'].add_(d_p)


        return loss

def get_optimizers(settings, classifier, alpha=None, s=None):
    optimizer_alpha = None

    # If lr scheduler is defined with lambda, the lr is multiplied with the initial lr
    if settings.lr_sched_type=='const' or settings.lr_sched_type=='resnet_custom_lr':
        init_lr = 1
    else:
        init_lr = settings.lr

    if settings.optimname=='sgd':
        optimizer_w = optim.SGD(classifier.parameters(), lr=init_lr)
        if alpha is not None: optimizer_alpha = optim.SGD([alpha], lr=s)
    elif settings.optimname=='omd':
        optimizer_w = OMD(classifier.parameters(), lr=init_lr)
        if alpha is not None: optimizer_alpha = OMD([alpha], lr=s)
    elif settings.optimname=='omdsgd':
        optimizer_w = OMD(classifier.parameters(), lr=init_lr)
        if alpha is not None: optimizer_alpha = optim.SGD([alpha], lr=s)
    elif settings.optimname=='momentum':
        optimizer_w = optim.SGD(classifier.parameters(), lr=init_lr, momentum=settings.momentum, nesterov=settings.nesterov)
        if alpha is not None: optimizer_alpha = optim.SGD([alpha], lr=s)
    elif settings.optimname=='adam':
        optimizer_w = optim.Adam(classifier.parameters(), lr=init_lr, betas=(settings.beta1, settings.beta2), eps=1e-08, weight_decay=0)
        if alpha is not None: optimizer_alpha = optim.SGD([alpha], lr=s)
    else:
        raise NotImplementedError
    
    return optimizer_w, optimizer_alpha

def resnet_custom_lr(epoch):
    if epoch<1:
        return 0.01
    elif epoch>=1 and epoch<80:
        return 0.1
    elif epoch>=80 and epoch<120:
        return 0.01
    elif epoch>=120:
        return 0.001
    

def get_lr_scheduler(settings, optimizer):
    if settings.lr_sched_type=='const':
        lr_scheduler = lambda epoch: settings.lr
        return LambdaLR(optimizer, lr_lambda=lr_scheduler)
    
    elif settings.lr_sched_type=='resnet_custom_lr':
        lr_scheduler = lambda epoch: resnet_custom_lr(epoch)
        return LambdaLR(optimizer, lr_lambda=lr_scheduler)
    
    elif settings.lr_sched_type=='decay':
        lr_scheduler = lambda epoch: (settings.lr_decay**epoch)
        
    elif settings.lr_sched_type=='cosine': ## only works with total iterations
        return CosineAnnealingLR(optimizer, settings.total_iterations)
    
    elif settings.lr_sched_type=='at_epoch':
        return MultiStepLR(optimizer, settings.lr_epoch_list, settings.lr_drop_factor)
    
    elif settings.lr_sched_type=='gammaT':
        lr_scheduler = lambda epoch: 1./((epoch+1)**settings.lr_gamma)
        return LambdaLR(optimizer, lr_lambda=lr_scheduler)
    
    elif settings.lr_sched_type=='dynamic':
        if settings.lr_monitor=='train_loss' or settings.lr_monitor=='val_loss':
            return ReduceLROnPlateau(optimizer, mode='min', factor=settings.lr_drop_factor, patience = settings.lr_patience, threshold_mode='rel', min_lr=settings.min_lr)
        elif settings.lr_monitor=='tr_acc' or settings.lr_monitor=='val_acc':
            return ReduceLROnPlateau(optimizer, mode='max', factor=settings.lr_drop_factor, patience = settings.lr_patience, threshold_mode='rel', verbose=True, min_lr=1e-8)
    else:
        raise NotImplementedError

    
    
