import torch
import numpy as np
from misc import VarToNumpy
import math
import pickle
from google.cloud import storage
import os
import torch.nn as nn

class Logger(object):
    def __init__(self, settings, num_trainpoints, trainbsz, num_batches, nb_epoch, classifier, store_weights, store_train_acc, store_val_acc, store_train_loss, store_val_loss, store_time):

        self.settings = settings
        self.store_weights = store_weights
        self.store_train_acc = store_train_acc
        self.store_val_acc = store_val_acc
        self.store_train_loss = store_train_loss
        self.store_val_loss = store_val_loss
        self.store_time = store_time

        self.num_acc_loss = int(math.ceil((math.ceil(float(num_trainpoints)/trainbsz)-1)/settings.eval_batch)+1)*nb_epoch
        if self.store_val_loss:
            self.total_val_loss = np.empty([self.num_acc_loss])
            self.val_loss_ind = 0
        if self.store_val_acc:
            self.total_val_acc = np.empty([self.num_acc_loss])
            self.val_acc_ind = 0
        if self.store_train_loss:
            self.total_train_loss = np.empty([self.num_acc_loss])
            self.train_loss_ind = 0
        if self.store_train_acc:
            self.total_train_acc = np.empty([self.num_acc_loss])
            self.train_acc_ind = 0
        if self.store_time:
            self.total_time = np.empty([self.num_acc_loss])
            self.time_ind = 0
    
    def log_acc_loss(self, train_acc, train_loss, val_acc, val_loss, time=None):
        if self.store_val_loss:
            self.total_val_loss[self.val_loss_ind] = val_loss
            self.val_loss_ind +=1
            
        if self.store_val_acc:
            self.total_val_acc[self.val_acc_ind] = val_acc
            self.val_acc_ind +=1
            
        if self.store_train_acc:
            self.total_train_acc[self.train_acc_ind] = train_acc
            self.train_acc_ind +=1

        if self.store_train_loss:
            self.total_train_loss[self.train_loss_ind] = train_loss
            self.train_loss_ind +=1

        if self.store_time:
            self.total_time[self.time_ind] = time
            self.time_ind +=1

class Logger_itr(object):
    def __init__(self, settings, num_trainpoints, trainbsz, num_batches, classifier, store_weights, store_train_acc, store_val_acc, store_train_loss, store_val_loss, store_time):

        self.settings = settings
        self.store_weights = store_weights
        self.store_train_acc = store_train_acc
        self.store_val_acc = store_val_acc
        self.store_train_loss = store_train_loss
        self.store_val_loss = store_val_loss
        self.store_time = store_time

        self.num_acc_loss = math.floor(settings.total_iterations/settings.eval_iter)+1
        if self.store_val_loss:
            self.total_val_loss = np.empty([self.num_acc_loss])
            self.val_loss_ind = 0
        if self.store_val_acc:
            self.total_val_acc = np.empty([self.num_acc_loss])
            self.val_acc_ind = 0
        if self.store_train_loss:
            self.total_train_loss = np.empty([self.num_acc_loss])
            self.train_loss_ind = 0
        if self.store_train_acc:
            self.total_train_acc = np.empty([self.num_acc_loss])
            self.train_acc_ind = 0
        if self.store_time:
            self.total_time = np.empty([self.num_acc_loss])
            self.time_ind = 0
    
    def log_acc_loss(self, train_acc, train_loss, val_acc, val_loss, time=None):
        if self.store_val_loss:
            self.total_val_loss[self.val_loss_ind] = val_loss
            self.val_loss_ind +=1
            
        if self.store_val_acc:
            self.total_val_acc[self.val_acc_ind] = val_acc
            self.val_acc_ind +=1
            
        if self.store_train_acc:
            self.total_train_acc[self.train_acc_ind] = train_acc
            self.train_acc_ind +=1

        if self.store_train_loss:
            self.total_train_loss[self.train_loss_ind] = train_loss
            self.train_loss_ind +=1

        if self.store_time:
            self.total_time[self.time_ind] = time
            self.time_ind +=1

def get_bucket(gcs_project_name, bucket_name):
    storage_client = storage.Client(project=gcs_project_name)
    bucket = storage_client.get_bucket(bucket_name)
    return bucket

def gsave(x, bucket, gsname):
    with open('tmp', "wb") as file:
        pickle.dump(x, file)
    tmp_blob = bucket.blob(gsname)
    tmp_blob.upload_from_filename('tmp')

def gsave_model(model, bucket, gsname):
    torch.save(model, 'tmp.pt')
    tmp_blob = bucket.blob(gsname)
    tmp_blob.upload_from_filename('tmp.pt')


class ConfusionLogger():
    def __init__(self, proj_name, wandb, gcs_root='logs'):
        self.wandb = wandb
        self.bucket = get_bucket('ml-theory', 'ybansal')
        self.gcs_logdir = f'{gcs_root}/{proj_name}/{wandb.run.id}'
        print("GCS Logdir:", self.gcs_logdir)

    def save(self, obj, ext):
        gsave(obj, self.bucket, f'{self.gcs_logdir}/{ext}')

    def log_setup(self, Y_tr, L_tr, Y_te, L_te):
        setup = {'Y_tr' : Y_tr,
                 'L_tr': L_tr,
                 'Y_te': Y_te,
                 'L_te' : L_te}

        self.save(setup, 'setup')

    def log_step(self, step, predsTr, predsTe, C_tr, C_te):
        prefix = f'steps/step{step:06}'
        self.save(predsTr, f'{prefix}/predsTr')
        self.save(predsTe, f'{prefix}/predsTe')
        self.save((C_tr, C_te), f'{prefix}/Cs')

    def log_final(self, step, predsTr, predsTe, C_tr, C_te):
        prefix = f'final'
        self.save(predsTr, f'{prefix}/predsTr')
        self.save(predsTe, f'{prefix}/predsTe')
        self.save((C_tr, C_te), f'{prefix}/Cs')
    

class DictLogger():
    def __init__(self, proj_name, wandb, gcs_root):
        ''' Logs run config, summmary, and history into GCS.
            For history: Appends logs to a local pickle file, and syncs this to GCS whenever sync() is called
        gcs_root here is the path within a bucket'''

        self.step = 0
        self.wandb = wandb
        self.bucket = get_bucket('ml-theory', 'ybansal')
        self.gcs_logdir =   f'{gcs_root}/{proj_name}/{wandb.run.id}' 
        print("GCS Logdir:", self.gcs_logdir)
        self.history = []
        self.gcs_history =   f'{self.gcs_logdir}/history'

        self._log_config()

    def _log_config(self):
        path = f'{self.gcs_logdir}/config'
        gsave(dict(self.wandb.config), self.bucket, path)

    def log(self, log_dict):
        log_dict['_step'] = self.step
        self.history.append(log_dict)
        self.step += 1

    def sync(self):
        gsave(self.history, self.bucket, self.gcs_history)
    
    def log_summary(self, log_dict):
        path = f'{self.gcs_logdir}/summary'
        gsave(log_dict, self.bucket, path)

    def log_activities(self, i, activities):
        path = f'{self.gcs_logdir}/activities/t_{i}'
        gsave(activities, self.bucket, path)

    def log_models(self, i, model):
        path = f'{self.gcs_logdir}/models/t_{i}'
        gsave_model(model, self.bucket, path)


###### Recording neuron ON/OFF

def get_preact_layer_inds(classifier):
    preact_layer_inds = []
    preac_layer_neurons = []
    
    for i in range(len(classifier)):
        if i<(len(classifier)-1) and isinstance(classifier[i+1], nn.ReLU):
            preact_layer_inds.append(i)
            
    return preact_layer_inds

class activity_recorder(object):
    def __init__(self, classifier):
        self.preact_layer_inds = get_preact_layer_inds(classifier) # Indices in CNN before ReLU
        self.neuron_activities = dict.fromkeys(np.arange(len(self.preact_layer_inds))) #bitarray(self.total_neurons*num_samples)
        self.register_layerwise_hooks(classifier)
        
    def log_bitarray(self, layer_ind):
        def make_bitarray_from_act(module, input, output):
            bits_string = ''.join(map(str, list((output.cpu()>0).view(-1).numpy())))
            bits_array = bitarray(bits_string)
            self.neuron_activities[layer_ind] = bits_array
        return make_bitarray_from_act
        
    def register_layerwise_hooks(self, classifier):
        for (i, layer_ind) in enumerate(self.preact_layer_inds):
            classifier[layer_ind].register_forward_hook(self.log_bitarray(i))

class norm_recorder(object):
    def __init__(self, depth, total_iterations):
        self.depth = depth
        self.weight_norm_history = np.empty(depth, total_iterations)
        self.bias_norm_history = np.empty(depth, total_iterations)

    def log_norms(self, classifier, iteration):
        
        self.norm_history
