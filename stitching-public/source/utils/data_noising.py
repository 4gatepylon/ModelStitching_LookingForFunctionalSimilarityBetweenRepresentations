import torch
import torch.nn as nn
import numpy as np
import numpy.random as npr
import copy
import sys

## Confusion matrix for CIFAR10 {ship, plane} -> 0 and {cat, dog} -> 1
cifar_spdc_binary = np.zeros((2, 10))
cifar_spdc_binary[0, 0] = 1.
cifar_spdc_binary[0, 8] = 1.
cifar_spdc_binary[1, 3] = 1.
cifar_spdc_binary[1, 5] = 1.

## Confusion matrix to remap {plane, ship, cat, dog} -> {0, 1, 2, 3}
cifar_spdc = np.zeros((4, 10))
cifar_spdc[0, 0] = 1.
cifar_spdc[1, 8] = 1.
cifar_spdc[2, 3] = 1.
cifar_spdc[3, 5] = 1.

# Confusion matrix to remap CIFAR10 to objects (0) v animals (1)
cifar_OA = np.zeros((2, 10))
cifar_OA[0, [0, 1, 8, 9]] = 1.
cifar_OA[1, [2, 3, 4, 5, 6, 7]] = 1.


class Flatten(nn.Module):
    def forward(self, x): return x.view(x.size(0), -1)


def reduce_train_size(train_dataset, numsamples):
    train_dataset.train_labels = train_dataset.train_labels[:numsamples]
    if 'train_data' in dir(train_dataset):
        train_dataset.train_data = train_dataset.train_data[:numsamples]
    else:
        train_dataset.train_filenames = train_dataset.train_filenames[:numsamples]
    if 'fine_labels' in dir(train_dataset): train_dataset.fine_labels = train_dataset.fine_labels[:numsamples]

def select_indices(dataset, new_labels, indices, train):
    if train:
        dataset.train_labels = new_labels
        dataset.train_labels = list(np.array(dataset.train_labels)[indices])
        dataset.train_data = dataset.train_data[indices]
        if 'fine_labels' in dir(dataset): dataset.fine_labels = list(np.array(dataset.fine_labels)[indices])
    else:
        dataset.test_labels = new_labels
        dataset.test_labels = list(np.array(dataset.test_labels)[indices])
        dataset.test_data = dataset.test_data[indices]
        if 'fine_labels' in dir(dataset): dataset.fine_labels = list(np.array(dataset.fine_labels)[indices])


def get_conf_matrix(settings, num_classes):

    if settings.noise_method=='fl_true_cm_unf':
    ## Each class mapped to a different class uniformaly with probability p
        p = settings.noise_probability
        confusion_matrix = (p/(num_classes-1))*np.ones((num_classes, num_classes))
        np.fill_diagonal(confusion_matrix, 1-p)

        No = num_classes
        Nfine = num_classes

    elif settings.noise_method=='fl_true_cm_binaryrandom':
        p = settings.noise_probability
        confusion_matrix = np.ones((2, num_classes))
        confusion_matrix[0, :] = ((1-p))*confusion_matrix[0, :]
        confusion_matrix[1, :] = (p)*confusion_matrix[1, :]

        No = 2
        Nfine = num_classes

    elif settings.noise_method=='fl_true_cm_binry_inc':
        confusion_matrix = np.vstack((np.linspace(0., .9, 10), 1-np.linspace(0., .9, 10)))

        No = 2
        Nfine = num_classes

    elif settings.noise_method=='fl_true_cm_supercat':
        confusion_matrix = cifar_OA
        p = settings.noise_probability
        confusion_matrix[1, 3] = 1-p
        confusion_matrix[0, 3] = p        

        No = 2
        Nfine = num_classes        

    elif settings.noise_method=='fl_true_cm_target':
    ## k -> k+1 with probability p. k is specified by settings.mislabel_target
        p = settings.noise_probability
        mislabel_target = settings.mislabel_target
        confusion_matrix = np.identity(num_classes)
        confusion_matrix[mislabel_target, mislabel_target] = 1-p
        confusion_matrix[(mislabel_target+1)%num_classes ,mislabel_target] = p

        No = num_classes
        Nfine = num_classes
        
    elif settings.noise_method=='fl_true_cm_super':
    ## i -> j//2 and noise j=0 to i=1 with probability p
        p = settings.noise_probability
        confusion_matrix = np.zeros((int(num_classes/2), num_classes))
        for i in range(num_classes):
            confusion_matrix[i/2, i] = 1.

        confusion_matrix[0, 0] = 1-p
        confusion_matrix[1, 0] = p

        No = num_classes
        Nfine = (num_classes-1)/2+1

    elif settings.noise_method=='fl_3true_cm_random':
        confusion_matrix = np.random.randint(0, 100, (3, 3))
        confusion_matrix = confusion_matrix/np.expand_dims(np.sum(confusion_matrix, axis=0), axis=0).repeat(3, 0)
        print(confusion_matrix)

        No = 3
        Nfine = 3

    elif settings.noise_method=='fl_true_cm_2random':
        confusion_matrix = 0.5*np.identity(10)
        for k in range(10):
            tmp = np.random.choice(9, 2, replace = False)
            tmp = (tmp<k)*tmp + (tmp>=k)*(tmp+1)
            confusion_matrix[tmp,k] = [0.3, 0.2]

        No = 10
        Nfine = 10
    
    elif settings.noise_method=='fl_cifar_RTspdc_cm_pcflip':
    ### {ship, plane} -> 0 and {cat, dog} -> 1. Then flip negative random teacher with prob p
        p = settings.noise_probability
        confusion_matrix = np.zeros((2, 4))
        confusion_matrix[0, 0] = p
        confusion_matrix[1, 0] = 1-p
        confusion_matrix[0, 1] = 1.

        confusion_matrix[1, 2] = p
        confusion_matrix[0, 2] = 1-p
        confusion_matrix[1, 3] = 1.

    else:
        raise NotImplementedError

    print(confusion_matrix)

    return confusion_matrix, No, Nfine


def get_new_labels(fine_labels, confusion_matrix):
    '''
    New labels are mapped according the confusion matrix. 
    Each column of a confusion matrix should sum to 1. or 0.
    If it sums to zero, those classes will be removed from 
    the dataset.
    '''
    new_labels = copy.deepcopy(fine_labels)
    new_labels = np.array(new_labels)
    indices = []
    column_sum = np.sum(confusion_matrix, axis=0)

    for j in range(confusion_matrix.shape[1]):
        if column_sum[j] != 0.:
            j_indices = list(np.where(np.array(fine_labels)==j)[0])
            new_labels[j_indices] = npr.choice(len(confusion_matrix[:, j]), len(j_indices), p=confusion_matrix[:, j])
            indices+= j_indices

    npr.shuffle(indices)

    return new_labels, indices

        

### Modify the datasets according to the required noising
def noise_dataset(settings, train_dataset, test_dataset):

    if settings.agreement:

        if settings.numsamples!=-1:
            raise ValueError("Current implementation assumes agreement is implemented with full dataset")

        split_size = settings.ag_split_size
        split_ind = settings.ag_split_ind
        
        beg = split_ind*split_size
        end = (split_ind+1)*split_size

        train_dataset.train_labels = train_dataset.train_labels[beg:end]
        train_dataset.train_data = train_dataset.train_data[beg:end]

    ### Noise that depends only on the true labels of the dataset
    if 'fl_true' in settings.noise_method:
        
        confusion_matrix, No, Nfine = get_conf_matrix(settings, num_classes=len(np.unique(train_dataset.train_labels)))

        ## Fine labels are the original classes
        train_dataset.fine_labels = copy.deepcopy(train_dataset.train_labels)
        test_dataset.fine_labels = copy.deepcopy(test_dataset.test_labels)

        ## New train labels according to the confusion matrix
        new_labels, indices = get_new_labels(train_dataset.fine_labels, confusion_matrix)
        new_test_labels, test_indices = get_new_labels(test_dataset.test_labels, confusion_matrix)

        ## Select data and labels according the the indices
        select_indices(train_dataset, new_labels, indices, train=True)
        select_indices(test_dataset, new_test_labels, test_indices, train=False)

    elif 'fl_finetune' in settings.noise_method:

        ## Fine labels are the original classes
        train_dataset.fine_labels = copy.deepcopy(train_dataset.train_labels)
        test_dataset.fine_labels = copy.deepcopy(test_dataset.test_labels)

        train_dataset.train_labels = train_dataset.train_labels[settings.ft_samples:]
        if 'train_data' in dir(train_dataset):
            train_dataset.train_data = train_dataset.train_data[settings.ft_samples:]
        else:
            train_dataset.train_filenames = train_dataset.train_filenames[settings.ft_samples:]

        ## Fine labels are the original classes
        train_dataset.fine_labels = copy.deepcopy(train_dataset.train_labels)
        test_dataset.fine_labels = copy.deepcopy(test_dataset.test_labels)

        confusion_matrix = np.eye(10)
        No = Nfine = 10
        
    elif 'fl_3true' in settings.noise_method:

        ## Fine labels are the original classes
        train_dataset.fine_labels = copy.deepcopy(train_dataset.train_labels)
        test_dataset.fine_labels = copy.deepcopy(test_dataset.test_labels)

        ## Get the first three classes only
        indices = [ind for ind in range(len(train_dataset.train_labels)) if train_dataset.train_labels[ind] in [0, 1, 2]]
        test_indices = [ind for ind in range(len(test_dataset.test_labels)) if test_dataset.test_labels[ind] in [0, 1, 2]]

        select_indices(train_dataset, train_dataset.train_labels, indices, train=True)
        select_indices(test_dataset, test_dataset.test_labels, test_indices, train=False)

        ## Apply random confusion matrix
        confusion_matrix, No, Nfine = get_conf_matrix(settings, num_classes=3)

        ## New train labels according to the confusion matrix
        new_labels, indices = get_new_labels(train_dataset.fine_labels, confusion_matrix)
        new_test_labels, test_indices = get_new_labels(test_dataset.test_labels, confusion_matrix)

        ## Select data and labels according the the indices
        select_indices(train_dataset, new_labels, indices, train=True)
        select_indices(test_dataset, new_test_labels, test_indices, train=False)

##########################################################################        
        
    elif settings.noise_method=='fl_subsample':
        p = np.linspace(0.1, 1., 10)

        train_dataset.fine_labels = [0]*len(train_dataset)
        test_dataset.fine_labels = [0]*len(test_dataset)

        train_indices = []
        ## Select for train
        for k in range(10):
            k_indices = list(np.where(np.array(train_dataset.train_labels)==k)[0])
            num = int(p[k]*len(k_indices))
            k_indices = np.random.choice(k_indices, size=num, replace=False)
            train_indices+=list(k_indices)

        select_indices(train_dataset, train_dataset.train_labels, train_indices, train=True)

        test_indices = []
        ## Select for test
        for k in range(10):
            k_indices = list(np.where(np.array(test_dataset.test_labels)==k)[0])
            num = int(p[k]*len(k_indices))
            k_indices = np.random.choice(k_indices, size=num, replace=False)
            test_indices+=list(k_indices)

        select_indices(test_dataset, test_dataset.test_labels, test_indices, train=False)

        No = 10
        Nfine = 1
        confusion_matrix = np.ones((10, 1))

##########################################################################

    elif settings.noise_method=='fl_splitdogs_cm_true':
        num_divs = settings.num_divs
        confusion_matrix = np.identity(10)
        cm2 = np.zeros((num_divs-1, 10))
        cm2[:,5] = 1./num_divs
        confusion_matrix[5, 5] = 1./num_divs
        confusion_matrix = np.concatenate((confusion_matrix, cm2), axis=0)

        No = num_divs + 9
        Nfine = 10

        ## Save true labels for debugging
        train_dataset.true_labels = copy.deepcopy(train_dataset.train_labels)
        test_dataset.true_labels = copy.deepcopy(test_dataset.test_labels)

        ## Fine labels are the original classes
        train_dataset.fine_labels = copy.deepcopy(train_dataset.train_labels)
        test_dataset.fine_labels = copy.deepcopy(test_dataset.test_labels)

        new_labels, indices = get_new_labels(train_dataset.fine_labels, confusion_matrix)
        new_test_labels, test_indices = get_new_labels(test_dataset.test_labels, confusion_matrix)

        ## Select data and labels according the the indices
        select_indices(train_dataset, new_labels, indices, train=True)
#        select_indices(test_dataset, new_test_labels, test_indices, train=False)

##########################################################################

    elif settings.noise_method=='fl_splitdogsall_cm_true':
        num_divs = settings.num_divs
        confusion_matrix = np.identity(10)
        cm2 = np.zeros((num_divs-1, 10))
        cm2[:,5] = 1./num_divs
        confusion_matrix[5, 5] = 1./num_divs
        confusion_matrix = np.concatenate((confusion_matrix, cm2), axis=0)

        No = num_divs + 9
        Nfine = 10

        ## Save true labels for debugging
        train_dataset.true_labels = copy.deepcopy(train_dataset.train_labels)
        test_dataset.true_labels = copy.deepcopy(test_dataset.test_labels)

        ## Fine labels are the original classes
        train_dataset.fine_labels = copy.deepcopy(train_dataset.train_labels)
        test_dataset.fine_labels = copy.deepcopy(test_dataset.test_labels)

        ## Set dog labels to unique index
        dog_inds = np.where(np.array(train_dataset.train_labels)==5)[0]
        a = np.arange(5000-1)
        np.random.shuffle(a)
        labels = np.array(train_dataset.train_labels)
        labels[dog_inds[1:]] = a+10

        train_dataset.train_labels = list(labels)


##########################################################################

    elif settings.noise_method=='fl_splitk_cm_true':

        k = settings.num_classes_split
        
        confusion_matrix = None
        No = 5000*k + 10
        Nfine = 10

        ## Save true labels for debugging
        train_dataset.true_labels = copy.deepcopy(train_dataset.train_labels)
        test_dataset.true_labels = copy.deepcopy(test_dataset.test_labels)

        ## Fine labels are the original classes
        train_dataset.fine_labels = copy.deepcopy(train_dataset.train_labels)
        test_dataset.fine_labels = copy.deepcopy(test_dataset.test_labels)

        ### Relabel all examples of first k classes as something else
        less_k = np.where(np.array(train_dataset.train_labels)<k)[0]
        geq_k = np.where(np.array(train_dataset.train_labels)>=k)[0]

        a = np.arange(5000*k)
        np.random.shuffle(a)
        labels = np.array(train_dataset.train_labels)
        labels[less_k] = a+10

        train_dataset.train_labels = list(labels)        

##########################################################################

    elif settings.noise_method=='fl_splitall_cm_true':
        confusion_matrix = None

        No = 50000
        Nfine = 10

        ## Save true labels for debugging
        train_dataset.true_labels = copy.deepcopy(train_dataset.train_labels)
        test_dataset.true_labels = copy.deepcopy(test_dataset.test_labels)

        ## Fine labels are the original classes
        train_dataset.fine_labels = copy.deepcopy(train_dataset.train_labels)
        test_dataset.fine_labels = copy.deepcopy(test_dataset.test_labels)

        train_dataset.train_labels = list(np.arange(50000))

        train_dataset.out_orig_map = np.array(train_dataset.true_labels)

##########################################################################

    elif settings.noise_method=='fl_splitallbin_cm_true':
        confusion_matrix = None

        No = 16
        Nfine = 10

        ## Save true labels for debugging
        train_dataset.true_labels = copy.deepcopy(train_dataset.train_labels)
        test_dataset.true_labels = copy.deepcopy(test_dataset.test_labels)

        ## Fine labels are the original classes
        train_dataset.fine_labels = copy.deepcopy(train_dataset.train_labels)
        test_dataset.fine_labels = copy.deepcopy(test_dataset.test_labels)

        train_dataset.train_labels = list(np.arange(50000))

        new_labels = np.empty((50000, 16), dtype=np.float32)
        for i in range(50000):
            new_labels[i] = [c for c in ("{:016b}".format(i))]

        train_dataset.train_labels = list(new_labels)
        train_dataset.out_orig_map = np.array(train_dataset.true_labels)

##########################################################################

    elif settings.noise_method=='fl_splitby2_cm_true':
        confusion_matrix = None

        ## Save true labels for debugging
        train_dataset.true_labels = copy.deepcopy(train_dataset.train_labels)
        test_dataset.true_labels = copy.deepcopy(test_dataset.test_labels)

        ## Fine labels are the original classes
        train_dataset.fine_labels = copy.deepcopy(train_dataset.train_labels)
        test_dataset.fine_labels = copy.deepcopy(test_dataset.test_labels)

        new_labels = np.arange(50000)
        label_inds = dict.fromkeys(np.arange(10))
        for k in range(10):
            label_inds[k] = np.where(np.array(train_dataset.train_labels)==k)[0]
    
            cnt = 0
            for j in label_inds[k][2500:]:
                new_labels[j] = label_inds[k][cnt]
                cnt += 1

        train_dataset.train_labels = list(new_labels)                

        No = max(new_labels)+1
        Nfine = 10


##########################################################################

    elif settings.noise_method=='fl_splitmbuckets_cm_true':
        confusion_matrix = None

        ## Save true labels for debugging
        train_dataset.true_labels = copy.deepcopy(train_dataset.train_labels)
        test_dataset.true_labels = copy.deepcopy(test_dataset.test_labels)

        ## Fine labels are the original classes
        train_dataset.fine_labels = copy.deepcopy(train_dataset.train_labels)
        test_dataset.fine_labels = copy.deepcopy(test_dataset.test_labels)

        new_labels = np.arange(50000)
        num_samples_per_class = 5000

        label_inds = dict.fromkeys(np.arange(10))
        for k in range(10):
            label_inds[k] = np.where(np.array(train_dataset.train_labels)==k)[0]

        samples_per_bucket = settings.samples_per_bucket
        split_size = int(num_samples_per_class/samples_per_bucket)

        for k in range(10):
            for i in range(samples_per_bucket):
                cnt = 0
                end = (i==(samples_per_bucket-1))*5000 + (i!=(samples_per_bucket-1))*(split_size*(i+1))
                for j in label_inds[k][split_size*i:end]:
                    new_labels[j] = label_inds[k][cnt]
                    cnt += 1        
        
        train_dataset.train_labels = list(new_labels)
        train_dataset.out_orig_map = np.array(train_dataset.true_labels[:max(new_labels)+1])

        No = max(new_labels)+1
        Nfine = 10

##########################################################################

    elif settings.noise_method=='fl_splitmnew_cm_true':
        confusion_matrix = None

        ## Save true labels for debugging
        train_dataset.true_labels = copy.deepcopy(train_dataset.train_labels)
        test_dataset.true_labels = copy.deepcopy(test_dataset.test_labels)

        ## Fine labels are the original classes
        train_dataset.fine_labels = copy.deepcopy(train_dataset.train_labels)
        test_dataset.fine_labels = copy.deepcopy(test_dataset.test_labels)

        numsamples = (settings.numsamples==-1)*len(train_dataset) + (settings.numsamples!=-1)*settings.numsamples
        new_labels = np.arange(numsamples)
        num_samples_per_class = np.unique(np.array(train_dataset.train_labels)[:numsamples], return_counts=True)[1]

        label_inds = dict.fromkeys(np.arange(10))
        for k in range(10):
            label_inds[k] = np.where(np.array(train_dataset.train_labels)==k)[0]

        samples_per_bucket = settings.samples_per_bucket

        for k in range(10):
            split_size = int(num_samples_per_class[k]/samples_per_bucket)
            for i in range(samples_per_bucket):
                cnt = 0
                end = (i==(samples_per_bucket-1))*num_samples_per_class[k] + (i!=(samples_per_bucket-1))*(split_size*(i+1))
                for j in label_inds[k][split_size*i:end]:
                    new_labels[j] = label_inds[k][cnt]
                    cnt += 1        
        
        train_dataset.train_labels = list(new_labels)
        train_dataset.out_orig_map = np.array(train_dataset.true_labels[:max(new_labels)+1])

        No = max(new_labels)+1
        Nfine = 10        

##########################################################################

    elif settings.noise_method=='fl_unsupindex_cm_true':
        confusion_matrix = None

        ## Save true labels for debugging
        train_dataset.true_labels = copy.deepcopy(train_dataset.train_labels)
        test_dataset.true_labels = copy.deepcopy(test_dataset.test_labels)

        ## Fine labels are the original classes
        train_dataset.fine_labels = copy.deepcopy(train_dataset.train_labels)
        test_dataset.fine_labels = copy.deepcopy(test_dataset.test_labels)

        n = settings.num_sup_samples
        new_labels = np.zeros(50000, dtype=np.int)
        new_labels[:n] = train_dataset.train_labels[:n]
        new_labels[n:] = np.arange(50000-n)+10

        output_orig_map = np.zeros(max(new_labels)+1, dtype=np.int)
        output_orig_map[:10] = np.arange(10)
        output_orig_map[10:] = train_dataset.train_labels[n:]

        train_dataset.train_labels = list(new_labels)
        train_dataset.out_orig_map = output_orig_map

        No = max(new_labels)+1
        Nfine = 10        


##########################################################################        

    elif 'fl_cifar_spdc' in settings.noise_method:

        ## Save true labels for debugging
        train_dataset.true_labels = copy.deepcopy(train_dataset.train_labels)
        test_dataset.true_labels = copy.deepcopy(test_dataset.test_labels)      

        ################# Get plane, ship, cat, dog ##################
        get_spdc_mat = cifar_spdc

        new_labels, indices = get_new_labels(train_dataset.train_labels, get_spdc_mat)
        new_test_labels, test_indices = get_new_labels(test_dataset.test_labels, get_spdc_mat)

        ## Select data and labels according the the indices
        select_indices(train_dataset, new_labels, indices, train=True)
        select_indices(test_dataset, new_test_labels, test_indices, train=False)

        ################ Fine labels are the original classes ##############
        train_dataset.fine_labels = copy.deepcopy(train_dataset.train_labels)
        test_dataset.fine_labels = copy.deepcopy(test_dataset.test_labels)          

        if 'cm_fine_check' in settings.noise_method:

            No = 4
            Nfine = 4
            confusion_matrix = np.identity(Nfine)

        elif 'cm_pflip' in settings.noise_method:

            No = 2
            Nfine = 4

            p = settings.noise_probability
            confusion_matrix = np.zeros((2, 4))
            confusion_matrix[0, 0] = 1-p
            confusion_matrix[1, 0] = p
            confusion_matrix[0, 1] = 1.
            confusion_matrix[1, 2] = 1
            confusion_matrix[1, 3] = 1.

            new_labels, indices = get_new_labels(train_dataset.fine_labels, confusion_matrix)
            new_test_labels, test_indices = get_new_labels(test_dataset.fine_labels, confusion_matrix)
            ## Select data and labels according the the indices
            select_indices(train_dataset, new_labels, indices, train=True)
            select_indices(test_dataset, new_test_labels, test_indices, train=False)            

        elif 'cm_pcflip' in settings.noise_method:

            No = 2
            Nfine = 4

            p = settings.noise_probability
            confusion_matrix = np.zeros((2, 4))
            confusion_matrix[0, 0] = 1-p
            confusion_matrix[1, 0] = p
            confusion_matrix[0, 1] = 1.
            confusion_matrix[0, 2] = p
            confusion_matrix[1, 2] = 1-p
            confusion_matrix[1, 3] = 1.

            new_labels, indices = get_new_labels(train_dataset.fine_labels, confusion_matrix)
            new_test_labels, test_indices = get_new_labels(test_dataset.fine_labels, confusion_matrix)
            ## Select data and labels according the the indices
            select_indices(train_dataset, new_labels, indices, train=True)
            select_indices(test_dataset, new_test_labels, test_indices, train=False)
            
        else:
            raise NotImplementedError

##########################################################################

    elif 'fl_cifar_OA' in settings.noise_method:

        ## Save true labels for debugging
        train_dataset.true_labels = copy.deepcopy(train_dataset.train_labels)
        test_dataset.true_labels = copy.deepcopy(test_dataset.test_labels)      

        if 'cm_fine_check' in settings.noise_method:

            ################# Map to objects vs animals ##################
            get_OA_mat = cifar_OA

            new_labels, indices = get_new_labels(train_dataset.train_labels, get_OA_mat)
            new_test_labels, test_indices = get_new_labels(test_dataset.test_labels, get_OA_mat)

            ## Select data and labels according the the indices
            select_indices(train_dataset, new_labels, indices, train=True)
            select_indices(test_dataset, new_test_labels, test_indices, train=False)

            ################ Fine labels are OA ##################
            train_dataset.fine_labels = copy.deepcopy(train_dataset.train_labels)
            test_dataset.fine_labels = copy.deepcopy(test_dataset.test_labels)          

            No = 2
            Nfine = 2
            confusion_matrix = np.identity(Nfine)

        elif 'cm_super_01' in settings.noise_method:

            p = settings.noise_probability

            ################ Fine labels are the original classes ##############
            train_dataset.fine_labels = copy.deepcopy(train_dataset.train_labels)
            test_dataset.fine_labels = copy.deepcopy(test_dataset.test_labels)              
            
            ################# Map to objects vs animals ##################
            get_OA_mat = cifar_OA

            new_labels, indices = get_new_labels(train_dataset.train_labels, get_OA_mat)
            new_test_labels, test_indices = get_new_labels(test_dataset.test_labels, get_OA_mat)

            ## Select data and labels according to the OA indices
            select_indices(train_dataset, new_labels, indices, train=True)
            select_indices(test_dataset, new_test_labels, test_indices, train=False)

            ## Change 8, 9 -> 6,7
            tmp = np.array(train_dataset.fine_labels)
            train_dataset.fine_labels = list((tmp>7)*(tmp-2)+(tmp<6)*tmp)

            tmp2 = np.array(test_dataset.fine_labels)
            test_dataset.fine_labels = list((tmp2>7)*(tmp2-2)+(tmp2<6)*tmp2)            

            confusion_matrix = cifar_OA[:,[0, 1, 2, 3, 4, 5, 8, 9]]
            confusion_matrix[1, 5] = 1-p
            confusion_matrix[0, 5] = p

            
            #### Add superclass noise
            new_labels, indices = get_new_labels(train_dataset.fine_labels, confusion_matrix)
            new_test_labels, test_indices = get_new_labels(test_dataset.fine_labels, confusion_matrix)
            
            ## Select data and labels according the the indices
            select_indices(train_dataset, new_labels, indices, train=True)
            select_indices(test_dataset, new_test_labels, test_indices, train=False)            

            No = 2
            Nfine = 8

        elif 'cm_oa_orig':

            ## Get the first three classes only
            indices = [ind for ind in range(len(train_dataset.train_labels)) if train_dataset.train_labels[ind] in [0, 1, 2, 3, 4, 5, 8, 9]]
            test_indices = [ind for ind in range(len(test_dataset.test_labels)) if test_dataset.test_labels[ind] in [0, 1, 2, 3, 4, 5, 8, 9]]

            select_indices(train_dataset, train_dataset.train_labels, indices, train=True)
            select_indices(test_dataset, test_dataset.test_labels, test_indices, train=False)

            ## Change 8, 9 -> 6,7
            tmp = np.array(train_dataset.train_labels)
            train_dataset.train_labels = list((tmp>7)*(tmp-2)+(tmp<6)*tmp)

            tmp2 = np.array(test_dataset.test_labels)
            test_dataset.test_labels = list((tmp2>7)*(tmp2-2)+(tmp2<6)*tmp2)

            train_dataset.fine_labels = [int(y in [0, 1, 6, 7]) for y in train_dataset.train_labels]
            test_dataset.fine_labels = [int(y in [0, 1, 6, 7]) for y in test_dataset.test_labels]

            No = 8
            Nfine = 2
            confusion_matrix = cifar_OA.T[[0, 1, 2, 3, 4, 5, 8, 9],:]

        else:
            raise NotImplementedError

##########################################################################            
        

    elif settings.noise_method == 'cub':

        confusion_matrix = np.identity(2)
        No = Nfine = 2

##########################################################################        


    elif 'fl_cifarbinary_RT' in settings.noise_method:

        print('rty')

        ## Save true labels for debugging
        train_dataset.true_labels = copy.deepcopy(train_dataset.train_labels)
        test_dataset.true_labels = copy.deepcopy(test_dataset.test_labels)

        ################# CIFAR Binary ###############
        cifar_binary_mat = np.zeros((2, 10))
        cifar_binary_mat[0, :5] = 1.
        cifar_binary_mat[1, 5:] = 1.

        new_labels, indices = get_new_labels(train_dataset.train_labels, cifar_binary_mat)
        new_test_labels, test_indices = get_new_labels(test_dataset.test_labels, cifar_binary_mat)

        ## Select data and labels according the the indices
        select_indices(train_dataset, new_labels, indices, train=True)
        select_indices(test_dataset, new_test_labels, test_indices, train=False)

        ################# Get fine labels from teacher ###############
        print('heyyy')
        train_rt_out, test_rt_out = get_random_teacher_output(train_dataset, test_dataset, settings)

        train_dataset.fine_labels = list(np.array(train_dataset.train_labels)*2+train_rt_out)
        test_dataset.fine_labels = list(np.array(test_dataset.test_labels)*2+test_rt_out)
        
        if 'cm_RTflip' in settings.noise_method:
            No = 2
            Nfine = 4

            ################ Noise by random teacher ###############
            p = settings.noise_probability
            confusion_matrix = np.zeros((2, 4))
            confusion_matrix[0, 0] = 1-p
            confusion_matrix[1, 0] = p
            confusion_matrix[0, 1] = 1.
            confusion_matrix[1, 2] = 1.
            confusion_matrix[1, 3] = 1.

        elif 'cm_RTcheck' in settings.noise_method:
            No = 4
            Nfine = 4
                
            confusion_matrix = np.identity(Nfine)

        elif 'cm_RTonlycheck' in settings.noise_method:

            train_dataset.fine_labels = list(np.array(train_rt_out, dtype=int))
            test_dataset.fine_labels = list(np.array(test_rt_out, dtype=int))            

            No = 2
            Nfine = 2

            confusion_matrix = np.identity(Nfine)

        new_labels, indices = get_new_labels(train_dataset.fine_labels, confusion_matrix)
        new_test_labels, test_indices = get_new_labels(test_dataset.fine_labels, confusion_matrix)

        ## Select data and labels according the the indices
        select_indices(train_dataset, new_labels, indices, train=True)
        select_indices(test_dataset, new_test_labels, test_indices, train=False)         

        
############################### OLD  ###############################          
            
    ### Noise that depends on a random teacher
    elif 'RT' in settings.noise_method:
        if 'fl_cifar_RTspdc' in settings.noise_method:
            
            ## Save true labels for debugging
            train_dataset.true_labels = copy.deepcopy(train_dataset.train_labels)
            test_dataset.true_labels = copy.deepcopy(test_dataset.test_labels)

            ################# Get (ship, plane) and (cat, dog) ###############
            get_spdc_mat = cifar_spdc_binary
            new_labels, indices = get_new_labels(train_dataset.train_labels, get_spdc_mat)
            new_test_labels, test_indices = get_new_labels(test_dataset.test_labels, get_spdc_mat)

            ## Select data and labels according the the indices
            select_indices(train_dataset, new_labels, indices, train=True)
            select_indices(test_dataset, new_test_labels, test_indices, train=False)
            
            ################# Get fine labels from teacher ###############
            print('heyyy')
            train_rt_out, test_rt_out = get_random_teacher_output(train_dataset, test_dataset, settings)
            train_dataset.fine_labels = list(np.array(train_dataset.train_labels)*2+train_rt_out)
            test_dataset.fine_labels = list(np.array(test_dataset.test_labels)*2+test_rt_out)

            if 'cm_RTflip' in settings.noise_method:
                No = 2
                Nfine = 4

                ################ Noise by random teacher ###############
                p = settings.noise_probability
                confusion_matrix = np.zeros((2, 4))
                confusion_matrix[0, 0] = p
                confusion_matrix[1, 0] = 1-p
                confusion_matrix[0, 1] = 1.
                confusion_matrix[0, 2] = 1-p
                confusion_matrix[1, 2] = p
                confusion_matrix[1, 3] = 1.

            elif 'cm_RTcheck' in settings.noise_method:
                No = 4
                Nfine = 4
                
                confusion_matrix = np.identity(Nfine)
                
            else:
                raise NotImplementedError

            new_labels, indices = get_new_labels(train_dataset.fine_labels, confusion_matrix)
            new_test_labels, test_indices = get_new_labels(test_dataset.fine_labels, confusion_matrix)

            ## Select data and labels according the the indices
            select_indices(train_dataset, new_labels, indices, train=True)
            select_indices(test_dataset, new_test_labels, test_indices, train=False)            
        elif settings.noise_method=='random_binary_teacher_linear_cifar10_stdc_check' or settings.noise_method=='random_binary_teacher_mCNN_k1_cifar10_stdc_check':
            No = 2
            Nfine = 2

            ## Save true labels for debugging
            train_dataset.true_labels = copy.deepcopy(train_dataset.train_labels)
            test_dataset.true_labels = copy.deepcopy(test_dataset.test_labels)
            

            ################# Get (ship, plane) and (cat, dog) ###############
            get_stdc_mat = np.zeros((2, 10))
            get_stdc_mat[0, 0] = 1  # plane -> 0
            get_stdc_mat[0, 8] = 1. # ship -> 0
            get_stdc_mat[1, 3] = 1. # cat -> 1
            get_stdc_mat[1, 5] = 1. # dog -> 1

            new_labels, indices = get_new_labels(train_dataset.train_labels, get_stdc_mat)
            new_test_labels, test_indices = get_new_labels(test_dataset.test_labels, get_stdc_mat)


            ## New train for binary task
            train_dataset.train_labels = new_labels
            train_dataset.train_labels = list(np.array(train_dataset.train_labels)[indices])
            train_dataset.train_data = train_dataset.train_data[indices]

            test_dataset.test_labels = new_test_labels
            test_dataset.test_labels = list(np.array(test_dataset.test_labels)[test_indices])
            test_dataset.test_data = test_dataset.test_data[test_indices]

            ################# Get fine labels ###############
            train_rt_out, test_rt_out = get_random_teacher_output(train_dataset, test_dataset, settings)
            train_dataset.fine_labels = list(np.array(train_dataset.train_labels)*0 + train_rt_out)
#            train_dataset.fine_labels = list(np.array(train_dataset.train_labels)*2 + train_rt_out)
#            train_dataset.fine_labels = list(np.array(train_dataset.train_labels))
            test_dataset.fine_labels = list(np.array(test_dataset.test_labels)*0 + test_rt_out)
#            test_dataset.fine_labels = list(np.array(test_dataset.test_labels)*2 + test_rt_out)
#            test_dataset.fine_labels = list(np.array(test_dataset.test_labels))

            train_dataset.train_labels = list(np.array(train_dataset.train_labels)*0 + train_rt_out)
#            train_dataset.train_labels = list(np.array(train_dataset.train_labels)*2 + train_rt_out)
            test_dataset.test_labels = list(np.array(test_dataset.test_labels)*0 + test_rt_out)
#            test_dataset.test_labels = list(np.array(test_dataset.test_labels)*0 + test_rt_out)

            confusion_matrix = np.identity(No)

        else:
            raise NotImplemented

    else:
        raise NotImplementedError

    if settings.numsamples!=-1:
        reduce_train_size(train_dataset, settings.numsamples)

    return confusion_matrix, No, Nfine


def get_random_teacher_output(train_dataset, test_dataset, settings):
    if settings.dlaas:
        sys.path.insert(0, 'source/models')
        sys.path.insert(0, 'source/utils')
    else:
        sys.path.insert(0, '../source/models')
        sys.path.insert(0, '../source/utils')
        
    from basicmodels import create_fc, create_cnn
    from resnet_pytorch_image_classification import resnet_pic
    from classifier_settings import initialize_scaled_kaiming
    from resnet import resnet18k_cifar

    #### Linear teacher
    if 'RT_linear_cifar10' in settings.noise_method:
        random_teacher = nn.Sequential(*[Flatten(), nn.Linear(32*32*3, 1)])
        # Zero mean it
        random_teacher[-1].bias.data = torch.zeros_like(random_teacher[-1].bias)
        random_teacher[-1].bias.data = -torch.mean(random_teacher(torch.Tensor(train_dataset.train_data))).data
        train_rt_out = ((random_teacher(torch.Tensor(train_dataset.train_data))>=0)[:,0].numpy())
        test_rt_out = ((random_teacher(torch.Tensor(test_dataset.test_data))>=0)[:,0].numpy())

    elif 'RT_simplelinear' in settings.noise_method:
        random_teacher = nn.Sequential(*[Flatten(), nn.Linear(32*32*3, 1)])
        # Simple weights
        random_teacher[-1].weight.data = torch.zeros_like(random_teacher[-1].weight)
        random_teacher[-1].weight.data[:, :32] = torch.ones_like(random_teacher[-1].weight[:, :32])
        # Zero mean it
        random_teacher[-1].bias.data = torch.zeros_like(random_teacher[-1].bias)
        random_teacher[-1].bias.data = -torch.mean(random_teacher(torch.Tensor(train_dataset.train_data))).data
        train_rt_out = ((random_teacher(torch.Tensor(train_dataset.train_data))>=0)[:,0].numpy())
        test_rt_out = ((random_teacher(torch.Tensor(test_dataset.test_data))>=0)[:,0].numpy())
        print(random_teacher)
        print(random_teacher[-1].weight.data)
        print(random_teacher[-1].bias.data)
        
    ##### mCNN_bn with k = 1
    elif 'RT_mCNN' in settings.noise_method:
        resolution = 32
        d_output = settings.numout
        c_in = 3
        c_first = settings.random_teacher_width
        n_layers = 5
        random_teacher = create_cnn(resolution, d_output, c_in, c_first, n_layers, False)
        class_init = initialize_scaled_kaiming(settings.weightscale, mode='fan_in', No=settings.numout, deep_linear=False)
        random_teacher.apply(class_init.initialize)

        # Zero mean it
        random_teacher[-1].bias.data = torch.zeros_like(random_teacher[-1].bias)
        random_teacher[-1].bias.data = -torch.mean(random_teacher(torch.Tensor(train_dataset.train_data).permute(0, 3, 1, 2))).data
        train_rt_out = ((random_teacher(torch.Tensor(train_dataset.train_data).permute(0, 3, 1, 2))>=0)[:,0].numpy())
        test_rt_out = ((random_teacher(torch.Tensor(test_dataset.test_data).permute(0, 3, 1, 2))>=0)[:,0].numpy())

        print('heyyyyyyyy')
        print(random_teacher[-1].weight.data)

    elif 'RT_resnet18k' in settings.noise_method:
        k = settings.random_teacher_width
        num_classes = 1
        random_teacher = resnet18k_cifar(k, num_classes)
        class_init = initialize_scaled_kaiming(settings.weightscale, mode='fan_in', No=num_classes, scale_bn=False, deep_linear=False)
        random_teacher.apply(class_init.initialize)

        random_teacher.eval()
        # Zero mean it
        random_teacher.linear.bias.data = torch.zeros_like(random_teacher.linear.bias)
        random_teacher.linear.bias.data = -torch.mean(random_teacher(torch.Tensor(train_dataset.train_data).permute(0, 3, 1, 2))).data
        train_rt_out = ((random_teacher(torch.Tensor(train_dataset.train_data).permute(0, 3, 1, 2))>=0)[:,0].numpy())
        test_rt_out = ((random_teacher(torch.Tensor(test_dataset.test_data).permute(0, 3, 1, 2))>=0)[:,0].numpy())        
        
    else:
        raise NotImplementedError

    return train_rt_out, test_rt_out
