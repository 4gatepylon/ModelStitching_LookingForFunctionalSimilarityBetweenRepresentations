from __future__ import print_function
import os, sys, time, math, argparse, torch
import numpy as np
import pickle as pkl

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

import torchvision.datasets as dsets
import torchvision.transforms as transforms

import wandb

#####################################################
################## Input Arg Def ####################
#####################################################

parser = argparse.ArgumentParser(fromfile_prefix_chars='@')

# Run configuration
parser.add_argument('--dlaas', action='store_true')
parser.add_argument('--rseed', type=int, default=-1)
parser.add_argument('--gpu', action='store_true')

# Dataset settings
parser.add_argument('--dataname', type=str, required=True, help='CIFAR10, CIFAR100, MNIST, ToyDataset')
parser.add_argument('--dataroot', help='Path to directory containing the data')
parser.add_argument('--transform_name', default='common', help='Type of transformations to the data. If None, it converts to tensor')
parser.add_argument('--corruption_ratio', type=float, help='Ratio of examples with wrong labels')
parser.add_argument('--train_inds_root', help='Path to training indices to be trained on')
parser.add_argument('--validation', action='store_true', help='Validate on part of training data') 
parser.add_argument('--numsamples', type=int, default=-1, help='Default state is testing, so all training samples are used for learning, if validation is True, this has to be less than total size')
parser.add_argument('--batchsize', type=int, default=-1)
parser.add_argument('--testbatchsize', type=int, required=True, help='Max size we can pass through the network')
parser.add_argument('--data_augment', action='store_true')

# Agreement
parser.add_argument('--agreement', action='store_true')
parser.add_argument('--ag_split_size', type=int, default=25000)
parser.add_argument('--ag_split_ind', type=int, help='Split 0 or 1')

# Stitching
parser.add_argument('--stitch_models', action='store_true')
parser.add_argument('--stitch_model1_path', type=str, help='Path to trained classifier 1 for stitching')
parser.add_argument('--stitch_model2_path', type=str, help='Path to trained classifier 2 for stitching')
parser.add_argument('--stitch_depth', type=int, default=1)
parser.add_argument('--stitch_layer_num', type=int)
parser.add_argument('--stitch_layer_num_top', type=int)
parser.add_argument('--stitch_kernel_size', type=int, default=1)
parser.add_argument('--stitch_no_bn', action='store_true')

# Corruption settings
parser.add_argument('--gnoise_std', type=float)

# Noise settings
parser.add_argument('--noise_method', type=str, help='Type of corruption to the dataset')
parser.add_argument('--noise_probability', type=float)
parser.add_argument('--mislabel_target', type=int)
parser.add_argument('--random_teacher_width', type=int)
parser.add_argument('--celeba_train_attr', type=str)
parser.add_argument('--celeba_fine_attr', type=str)
parser.add_argument('--num_divs', type=int)
parser.add_argument('--num_classes_split', type=int)

# Number of worker settings 
parser.add_argument('--tr_workers', type=int, default=2) 
parser.add_argument('--val_workers', type=int, default=2) 
parser.add_argument('--tr_val_workers', type=int, default=2) 
parser.add_argument('--test_workers', type=int, default=2) 

# Classifier settingsda
parser.add_argument('--classifierpath', type=str, help='Path to trained classifier if loading saved classifier') 
parser.add_argument('--classifiername', type=str)
parser.add_argument('--numout', type=int, required=True, help='Number of output classes')
parser.add_argument('--weightscale', type=float, default=1., help='For FC Networks: Scale of weight init (for precise definition see model file)')
parser.add_argument('--add_batchnorm', action='store_true')
parser.add_argument('--scale_bn', action='store_true')
parser.add_argument('--unfreeze_layers', default=-1, type=int, help='Number of layers from the top to keep trainable')
parser.add_argument('--freeze_layers', default=-1, type=int, nargs='*', help='Number of layers from the bottom to keep untrainable')

#### Classifier specific options
# Fully Connected Network
parser.add_argument('--numinput', type=int, help='For FC Networks: Number of input dimensions') 
parser.add_argument('--numhid', type=int, help='For FC Networks: Number of hidden units (Assumed identical number for each hidden layer)') 
parser.add_argument('--depth', type=int, help='For FC Networks: Number of hidden layers')

# convnet MNIST
parser.add_argument('--conv_channels', type=int)

# Optimizer
parser.add_argument('--optimname', type=str, help='sgd, momentum, adam')
parser.add_argument('--momentum', type=float, help='Momentum for SGD+momentum')
parser.add_argument('--nesterov', action='store_true')
parser.add_argument('--beta1', type=float, help='beta1 for Adam')
parser.add_argument('--beta2', type=float, help='beta2 for Adam')

parser.add_argument('--weight_decay_coeff', type=float)
parser.add_argument('--losstype', type=str, default='ce', help='mse, ce')

# Training Settings
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--total_iterations', type=int)
parser.add_argument('--tr_loss_thresh', type=float, default=1e-6)
parser.add_argument('--lr_sched_type', help='const, at_epoch, dynamic')
parser.add_argument('--lr', type=float, help='const, at_epoch, dynamic: init LR value')
parser.add_argument('--lr_decay', type=float, help='decay: Factor to multiply LR by')
parser.add_argument('--lr_epoch_list', type=int, default = 0, nargs='*', help='at_epoch: Epochs to change the learning rate to value from previous argument')
parser.add_argument('--lr_drop_factor', type=float, help='at_epoch, dynamic: lr decay factor for MultiStepLR')
parser.add_argument('--lr_patience', type=int, help='dynamic: Number of updates to wait before dropping LR')
parser.add_argument('--lr_monitor', type=str, help='dynamic: Quantity to monitor for LR updates (train_acc, train_loss, val_acc, val_loss)')
parser.add_argument('--min_lr', type=float, default=1e-8, help=' dynamic: minimum LR for dynamic LR schedule')
parser.add_argument('--lr_gamma', type=float)


# Data gathering settings
parser.add_argument('--eval_iter', type=int, default=500, help='Record validation accuracies after these many iterations')
parser.add_argument('--record_activities', action='store_true')
parser.add_argument('--log_predictions', action='store_true')
parser.add_argument('--showplot', action='store_true')
parser.add_argument('--saveplot', dest='saveplot', action='store_true')
parser.add_argument('--savefile', action='store_true')
parser.add_argument('--savepath', type=str)
parser.add_argument('--gcs_root', type=str)
parser.add_argument('--gcs_key_path', type=str)
parser.add_argument('--verbose', action='store_true')

parser.add_argument('--save_iters', type=int, nargs='*')

# wandb settings
parser.add_argument('--wandb_project_name', type=str)
parser.add_argument('--wandb_api_key_path', type=str)

settings = parser.parse_args();
print(settings)
print(settings.save_iters)
# wandb configure
if settings.dlaas:
    wandb_path = os.path.join('source', settings.wandb_api_key_path)
else:
    wandb_path = os.path.join(settings.wandb_api_key_path)
    
    with open(wandb_path, 'r') as f:
        key = f.read().split('\n')[0]
    os.environ['WANDB_API_KEY'] = key

wandb.init(project=settings.wandb_project_name, entity='yaminibansal')
wandb.config.update(settings)
wandb_run_id = wandb.run.get_url().split('/')[-1]

if settings.classifiername=='FC_linear' or settings.classifiername=='FC_relu' or settings.classifiername=='FC_linear_bias' or settings.classifiername=='FC_relu_bias':
    if settings.numinput is None or settings.numhid is None or settings.numout is None or settings.depth is None: 
        parser.error('For FC networks, numinput, numhid, depth and numout are required')

if settings.optimname=='momentum':
    if settings.momentum is None:
        parser.error('For momentum optimizer, please specify momentum value')

if settings.optimname=='adam':
    if settings.beta1 is None or settings.beta2 is None:
        parser.error('For ADAM optimizer, please specify beta1 and beta2 values')

if settings.lr_sched_type=='const':
    if not settings.lr:
        parser.error('For constant LR, please specify learning rate lr')

if settings.lr_sched_type=='decay':
    if not settings.lr or not settings.lr_decay:
        parser.error('For decay LR, please specify learning rate lr and decay lr_decay')

if settings.lr_sched_type=='at_epoch':
    #if not settings.lr_list or not settings.lr_epoch_list or not settings.lr_mult_factor:
    if not settings.lr_epoch_list or not settings.lr_drop_factor:
        parser.error('For at_epoch LR scheduler, please specify lr_list, lr_epoch_list and lr_mult_factor')

if settings.lr_sched_type=='dynamic':
    if not settings.lr_monitor or not settings.lr_patience or not settings.lr_drop_factor:
        parser.error('For dynamic LR, please specify lr_monitor, lr_patience and lr_drop_factor')

#####################################################
#################### Configure ######################
#####################################################

# Random seed settings
if settings.rseed!=-1:
    rseed = settings.rseed
else:
    rseed = np.random.randint(10000)
print('random seed = %d'%(rseed))
np.random.seed(rseed)
torch.manual_seed(rseed)
if settings.gpu:
    torch.cuda.manual_seed(rseed)


if settings.dlaas is True:
    # Imports
    sys.path.insert(0, 'source/utils')
    if settings.dataroot=='DATA_DIR':
        data_dir = os.environ["DATA_DIR"]
    else:
        data_dir = settings.dataroot
    results_dir = os.environ["RESULT_DIR"]
    print('Results store at %s'%results_dir)    

    if settings.classifierpath is not None:
        classifierpath = os.path.join('/'.join(results_dir.split('/')[:-1]), settings.classifierpath)
    else:
        classifierpath = None

else:
    sys.path.insert(0, '../source/utils')
    data_dir = settings.dataroot
    results_dir = os.path.join(settings.savepath, settings.wandb_project_name, wandb_run_id)

    classifierpath = settings.classifierpath

    try:
        os.makedirs(results_dir)
    except OSError:
        pass

    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = settings.gcs_key_path
    
wandb.run.summary[f"results_dir"] = results_dir

if settings.gcs_root is not None:
    dict_logger_root = settings.gcs_root
else:
    dict_logger_root = results_dir
    
if settings.showplot or settings.saveplot:
    import matplotlib

    if not settings.showplot:
        matplotlib.use('Agg')
    import matplotlib.pyplot as plt

from classifier_settings import get_classifier
from data_settings import get_dataset, make_dataloaders
from baseline_settings import get_baseline_loss
from optims_lr import get_optimizers, get_lr_scheduler
from misc import VarToNumpy, sign_acc_fn, timeSince, const_val_epoch
from plotting import plot_confusion_matrix
from logger import Logger, Logger_itr, DictLogger, activity_recorder, ConfusionLogger
from train_functions import evaluate, evaluate_noisy, evaluate_confusion_matrix
from data_noising import noise_dataset

# Print settings
print(settings)

#####################################################
###################### Dataset ######################
#####################################################

train_dataset, test_dataset = get_dataset(settings.dataname, data_dir, settings.dlaas, settings.classifiername, settings.data_augment, settings.transform_name, settings.corruption_ratio, settings)

confusion_matrix, No, Nfine = noise_dataset(settings, train_dataset, test_dataset) 

ds_tr_conf_mat = evaluate_confusion_matrix(train_dataset, torch.Tensor(train_dataset.train_labels), No, Nfine)
ds_te_conf_mat = evaluate_confusion_matrix(test_dataset, torch.Tensor(test_dataset.test_labels), No, Nfine)
print(f'True/Fine confusion matrix {ds_tr_conf_mat}')

(tr_data_loader, tr_data_loader_for_val, val_data_loader, test_data_loader, num_trainpoints, trainbsz, num_testpoints, testbsz) = make_dataloaders(settings.numsamples, No, settings.batchsize, settings.testbatchsize, train_dataset, test_dataset, settings.validation, shuffle=False, tr_workers=settings.tr_workers, val_workers=settings.val_workers, test_workers=settings.test_workers, tr_val_workers=settings.tr_val_workers, train_inds_root=settings.train_inds_root)
if settings.savefile and settings.corruption_ratio is not None:
    np.save(os.path.join(results_dir, "corrupt_indices.npy"), train_dataset.corrupt_indices)
    np.save(os.path.join(results_dir, "train_labels.npy"), train_dataset.train_labels)
print('Size of dataset = %d'%num_trainpoints)

#####################################################
#################### Classifier #####################
#####################################################
class Flatten(nn.Module):
    def forward(self, x): return x.view(x.size(0), -1)

classifier = get_classifier(settings, No, classifierpath)
if settings.gpu: classifier = classifier.cuda()
if settings.savefile:
    torch.save(classifier, os.path.join(results_dir, "init_model.pt"))

##############################################################
#################### Initialize Variables ####################
##############################################################
best_acc = 0
best_model_path = os.path.join(results_dir, "best_model.pt")
num_batches = int(math.ceil(float(num_trainpoints)/trainbsz))

loss_fn = get_baseline_loss(settings)

logger = Logger_itr(settings, num_trainpoints, trainbsz, num_batches, classifier, store_weights=False, store_train_acc=True, store_val_acc=True, store_train_loss=True, store_val_loss=True, store_time=True)

conf_logger = ConfusionLogger(settings.wandb_project_name, wandb)
conf_logger.log_setup(train_dataset.train_labels, train_dataset.fine_labels, test_dataset.test_labels, test_dataset.fine_labels)

dict_logger = DictLogger(settings.wandb_project_name, wandb, dict_logger_root)
if settings.record_activities: act_recorder = activity_recorder(classifier)

def update_summary(metrics):
    for k, v in metrics.items():
        wandb.run.summary[f"Final {k}"] = v

##############################################################
############### Learning Rate and Optimizer ##################
##############################################################
optimizer_w, _ = get_optimizers(settings, classifier) #initial_lr = 1.
lr_scheduler = get_lr_scheduler(settings, optimizer_w)

for param_group in optimizer_w.param_groups:
    if settings.weight_decay_coeff is not None: param_group['weight_decay'] = settings.weight_decay_coeff

##############################################################
####################### Begin training #######################
##############################################################
tic = time.time()
iterations = 0
tr_loss = 1e6

while(iterations<settings.total_iterations and tr_loss>settings.tr_loss_thresh):
    
    if settings.lr_sched_type != 'dynamic' or settings.lr_sched_type != 'cosine':
        lr_scheduler.step()

    for (i, X) in enumerate(tr_data_loader):
        classifier.train()
        
        input = Variable(X[0])
        target = Variable(X[1])
        data_inds = X[2] #torch.LongTensor
        if settings.gpu:
            input = input.cuda()
            target = target.cuda()
            data_inds = X[2].cuda()
        prediction = classifier(input)

        optimizer_w.zero_grad()
        loss_w = loss_fn(target, prediction)
        
        loss_w.backward()
        optimizer_w.step()

        if iterations%settings.eval_iter==0 and iterations<settings.total_iterations:
            (val_prediction, val_loss, val_sign_acc) = evaluate(val_data_loader, settings.gpu, classifier, loss_fn, sign_acc_fn, True)
            (tr_prediction, tr_loss, tr_sign_acc) = evaluate(tr_data_loader_for_val, settings.gpu, classifier, loss_fn, sign_acc_fn, True)
            tr_conf_mat = evaluate_confusion_matrix(train_dataset, tr_prediction, No, Nfine)
            te_conf_mat = evaluate_confusion_matrix(test_dataset, val_prediction, No, Nfine)
            test_tv = np.sum(np.abs(te_conf_mat - ds_te_conf_mat))/2
            
            conf_logger.log_step(iterations, tr_prediction, val_prediction, tr_conf_mat, te_conf_mat)
            
            if settings.corruption_ratio is not None:
                (tr_prediction_noisy, tr_loss_noisy, tr_sign_acc_noisy) = evaluate_noisy(tr_data_loader_for_val, settings.gpu, classifier, train_dataset.corrupt_indices, loss_fn, sign_acc_fn, settings.log_predictions)
                
            logger.log_acc_loss(tr_sign_acc, tr_loss, val_sign_acc, val_loss, time.time()-tic)
            print('I = %d, B = %d, time = %s, tr acc = %f, tr_loss = %f, te acc=%f, te_loss = %f'%(iterations, i, timeSince(tic), tr_sign_acc, tr_loss, val_sign_acc, val_loss))

            ### wandb and new logger logging
            for param_group in optimizer_w.param_groups:
                current_learning_rate = param_group['lr']

            metrics = {'Step' : iterations+1}
            metrics["Train Loss"] = tr_loss
            metrics["Train Error"] = 1.0-tr_sign_acc
            metrics["Test Loss"] = val_loss
            metrics["Test Error"] = 1.0-val_sign_acc
            metrics["Gen Error"] = tr_sign_acc - val_sign_acc
            metrics["TV distance"] = test_tv
            metrics['lr'] = current_learning_rate

            if settings.corruption_ratio is not None:
                metrics["Noisy Train Loss"] = tr_loss_noisy
                metrics["Noisy Train Error"] = 1.0-tr_sign_acc_noisy    
                
            wandb.log(metrics)
            update_summary(metrics)

            tr_cm_fig, _ = plot_confusion_matrix(tr_conf_mat)
            te_cm_fig, _ = plot_confusion_matrix(te_conf_mat)
            
#            wandb.log({'C_tr': [wandb.Image(tr_cm_fig, caption='C_tr'+str(iterations))],
#                      'C_te': [wandb.Image(te_cm_fig, caption='C_te'+str(iterations))]})
            
            if settings.log_predictions: metrics['Test Predictions'] = val_prediction.numpy()
            if settings.log_predictions: metrics['Train Predictions'] = tr_prediction.numpy()
            dict_logger.log(metrics)
            dict_logger.log_summary(metrics)
            dict_logger.sync()
            if settings.record_activities: dict_logger.log_activities(act_recorder.neuron_activities)

            if val_sign_acc > best_acc and settings.savefile:
                best_acc = val_sign_acc
                print('Saving new best model...')
                torch.save(classifier, best_model_path)
#                (best_test_prediction, best_test_loss, best_test_sign_acc) = evaluate(test_data_loader, settings.gpu, classifier, loss_fn, sign_acc_fn)
#                print('Current Best: Train Accuracy: %f, Val Accuracy: %f, Test Accuracy: %f'%(tr_sign_acc, val_sign_acc, best_test_sign_acc))



        if settings.lr_sched_type=='dynamic':
            if settings.lr_monitor=='train_loss':
                lr_scheduler.step(loss_w)
            elif settings.lr_monitor=='val_loss':
                lr_scheduler.step(val_loss)
            elif settings.lr_monitor=='tr_acc':
                lr_scheduler.step(tr_sign_acc)
            elif settings.lr_monitor=='val_acc':
                lr_scheduler.step(val_sign_acc)
            else:
                raise NotImplementedError
        if settings.lr_sched_type=='cosine':
            lr_scheduler.step()

        if (settings.stitch_models or settings.freeze_layers!=-1) and iterations+i==0 :
            for i in range(len(classifier)):
                for param in classifier[i].parameters():
                    print(param.requires_grad)

        if settings.save_iters is not None and iterations in settings.save_iters:
            torch.save(classifier, os.path.join(results_dir, f"model_{iterations}.pt"))

        iterations += 1


(test_prediction, test_loss, test_sign_acc) = evaluate(test_data_loader, settings.gpu, classifier, loss_fn, sign_acc_fn)
print('Final Test Accuracy: %f'%test_sign_acc)

(val_prediction, val_loss, val_sign_acc) = evaluate(val_data_loader, settings.gpu, classifier, loss_fn, sign_acc_fn, True)
(tr_prediction, tr_loss, tr_sign_acc) = evaluate(tr_data_loader_for_val, settings.gpu, classifier, loss_fn, sign_acc_fn, True)
#tr_conf_mat = evaluate_confusion_matrix(train_dataset, tr_prediction, No, Nfine)
#te_conf_mat = evaluate_confusion_matrix(test_dataset, val_prediction, No, Nfine)

#conf_logger.log_final(iterations, tr_prediction, val_prediction, tr_conf_mat, te_conf_mat)
#conf_logger.save(classifier.state_dict(), 'final/model') # optional
            

##############################################################
########################## Save and plot #####################
##############################################################
if settings.savefile:

    torch.save(classifier, os.path.join(results_dir, "final_model.pt"))
    np.save(os.path.join(results_dir, "rseed.npy"), rseed)
    np.save(os.path.join(results_dir, "params.npy"), settings)
    if logger.store_train_loss: np.save(os.path.join(results_dir,'train_loss.npy'), logger.total_train_loss)
    if logger.store_train_acc: np.save(os.path.join(results_dir,'train_acc.npy'), logger.total_train_acc)
    if logger.store_val_loss: np.save(os.path.join(results_dir,'val_loss.npy'), logger.total_val_loss)
    if logger.store_val_acc: np.save(os.path.join(results_dir,'val_acc.npy'), logger.total_val_acc)
    if logger.store_time: np.save(os.path.join(results_dir,'time.npy'), logger.total_time)
    np.save(os.path.join(results_dir,'test_result.npy'), (test_loss, test_sign_acc))
