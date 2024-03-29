import pickle as pkl
import sys
import os
import numpy as np
import time

# Add our source directory to the path as Python doesn't like sub-directories
source_path = os.path.join("..", "dlaas_source")
sys.path.insert(0, source_path)
from grid_search import GridSearch

rseed_list = list(np.random.randint(0, 10000, 1))
print(rseed_list)

s = '''#!/bin/bash

#SBATCH -p cox
#SBATCH --gres=gpu:1
#SBATCH --mem=20000
#SBATCH -t 800
#SBATCH -o /n/coxfs01/ybansal/sbatchfiles/noise_ens/check%j.out
#SBATCH -e /n/coxfs01/ybansal/sbatchfiles/noise_ens/check%j.err

module load Anaconda3/5.0.1-fasrc02
module load cuda/10.0.130-fasrc01 cudnn/7.4.1.5_cuda10.0-fasrc01

source activate repgen_pt

cd /n/coxfs01/ybansal/main-exp/hvd-notebooks/

{0}

exit 0;
'''

stitch_indices = {
    'resnet18k_cifar': [1, 2, 3, 4, 5, 6, 7, 8, 9, 11] # 10 is the output of average pooling, 11 stitcher should be an FC
}

model_desc_numsamples = {
    'resnet18_5K_init1': '/n/coxfs01/ybansal/main-exp/results/rep-compare-models/96jnwblw/final_model.pt',
    'resnet18_5K_init2': '/n/coxfs01/ybansal/main-exp/results/rep-compare-models/8v7n7wjc/final_model.pt',
    'resnet18_10K_init1': '/n/coxfs01/ybansal/main-exp/results/rep-compare-models/f1rkor6g/final_model.pt',
    'resnet18_10K_init2': '/n/coxfs01/ybansal/main-exp/results/rep-compare-models/3db6566h/final_model.pt',
    'resnet18_25K_init1': '/n/coxfs01/ybansal/main-exp/results/rep-compare-models/ikmzzd9h/final_model.pt',
    'resnet18_25K_init2': '/n/coxfs01/ybansal/main-exp/results/rep-compare-models/ge9ahm3b/final_model.pt'
}

model_desc_noise = {
    'resnet18_p005_init1': '/n/coxfs01/ybansal/main-exp/results/rep-compare-models/q70wk9qn/final_model.pt',
    'resnet18_p01_init1': '/n/coxfs01/ybansal/main-exp/results/rep-compare-models/g2p3wku5/final_model.pt',
    'resnet18_p05_init1': '/n/coxfs01/ybansal/main-exp/results/rep-compare-models/0n1h919o/final_model.pt',
}

model_bad_cka = {
    'resnet38_10x_init1': '/n/coxfs01/ybansal/main-exp/rep-compare-models/1ayqv8zl/final_model.pt',
    'resnet38_10x_init2': '/n/coxfs01/ybansal/main-exp/rep-compare-models/nwg7094q/best_model.pt',
    'resnet164_init1': '/n/coxfs01/ybansal/main-exp/rep-compare-models/3k3tfylg/final_model.pt', 
    'resnet164_init2': '/n/coxfs01/ybansal/main-exp/rep-compare-models/28jwcwc3/final_model.pt'
}

model_desc = {
    'resnet20_init1': '/n/coxfs01/ybansal/main-exp/results/rep-compare-models/lyuk8n5i/final_model.pt',
    'resnet20_init2': '/n/coxfs01/ybansal/main-exp/results/rep-compare-models/5m01yt5q/final_model.pt',
    'resnet18_init1': '/n/coxfs01/ybansal/main-exp/results/rep-compare-models/anxwa5og/final_model.pt',
    'resnet18_init2': '/n/coxfs01/ybansal/main-exp/results/rep-compare-models/wbympzd6/final_model.pt',
    'resnet18_oa_init1': '/n/coxfs01/ybansal/main-exp/results/rep-compare-models/1riyo69g/final_model.pt',
    'resnet18_oa_init2': '/n/coxfs01/ybansal/main-exp/results/rep-compare-models/caf6r2z8/final_model.pt',
    'resnet110_init1': '/n/coxfs01/ybansal/main-exp/results/rep-compare-models/9yh7pfxs/final_model.pt',
    'resnet110_init2': '/n/coxfs01/ybansal/main-exp/results/rep-compare-models/45r64ug4/final_model.pt',
    'cf100_resnet20_init1': '/n/coxfs01/ybansal/main-exp/results/rep-compare-models/8r0w6hzv/final_model.pt',
    'cf100_resnet110_init1': '/n/coxfs01/ybansal/main-exp/results/rep-compare-models/djun6l6g/final_model.pt',
    'resnet18_simclr_init1': '/n/coxfs01/ybansal/pretrain-rep/CIFAR-10/simclr_resnet18_28w562hk/28w562hk/model.pt',
    'resnet18_simclr_init2': '/n/coxfs01/ybansal/pretrain-rep/CIFAR-10/simclr_resnet18_28w562hk/28w562hk/last_layer_trained_model.pt'
}

project_name = sys.argv[1]

def sgd_const(i, j):
    search = GridSearch()
    search.add_list("rseed", rseed_list)
    search.add_static_var("gpu", '')

    search.add_static_var("dataname", 'CIFAR10withInds')
    search.add_static_var("dataroot", '/n/coxfs01/ybansal/Datasets/CIFAR10')
    search.add_static_var("data_augment", '')
    search.add_static_var("numsamples", -1) 
    search.add_list("batchsize", [128])
    search.add_static_var("testbatchsize", 128)

    search.add_static_var("noise_method", 'fl_true_cm_unf')
    search.add_static_var("noise_probability", 0.)    

    search.add_static_var("tr_workers", 0)
    search.add_static_var("val_workers", 0)
    search.add_static_var("test_workers", 0)
    search.add_static_var("tr_val_workers", 0)

    search.add_static_var("classifiername", 'resnet18k_cifar')
    search.add_list("conv_channels", [64])

    search.add_static_var("stitch_models", '')
    search.add_list("stitch_model1_path", [model_desc['resnet18_init1']])
    search.add_list("stitch_model2_path", [model_desc['resnet18_init2']])#, model_desc['resnet18_init2']])
    search.add_list("stitch_layer_num", [i])
    search.add_list("stitch_layer_num_top", [j])
    search.add_list("stitch_kernel_size", [1])
    search.add_list("stitch_depth", [1])
#    search.add_list("stitch_no_bn", '')

    search.add_static_var("numout", 10)
    search.add_static_var("weightscale", 1.0)

    search.add_static_var("optimname", 'adam')
    search.add_static_var("losstype", 'ce')

    search.add_list("lr_sched_type", ['cosine'])
    search.add_list("lr", [0.001])
    search.add_static_var("beta1", 0.9)
    search.add_static_var("beta2", 0.999)

    search.add_static_var("total_iterations", 120000)
    search.add_static_var("tr_loss_thresh", 1e-6)

    search.add_static_var("wandb_project_name", 'stitch-cf10-r18-diff-depth-k1')    
    
    search.add_static_var("eval_iter", 4000)
    search.add_static_var("verbose", '')
    search.add_static_var("saveplot", '')
    search.add_static_var("savefile", '')
    search.add_static_var("savepath", '..')
    search.add_static_var("gcs_root", 'logs')
    search.add_static_var("wandb_api_key_path", '/n/coxfs01/ybansal/settings/wandb_api_key.txt')
    search.add_static_var("gcs_key_path", '/n/coxfs01/ybansal/settings/ML-theory-b1e9f085f023.json')


    return search.create_grid_search()

run_dict = []

for i in range(1, 10):
    for j in range(1, 10):
        top = 11 if j==10 else j
#        print(i, top)
        runs = sgd_const(i, top)

        for index, run_params in enumerate(runs):
            command = "python3 ../source/noise_ens_dev.py"
            for name in run_params:
                command = "%s --%s %s" % (command, name, str(run_params[name]))
            slurm_file_in = s.format(command)
            with open('tmp.slurm', 'w') as f:
                f.write(slurm_file_in)
            term_out = os.system('sbatch tmp.slurm')

            # Store run commands and slurm output
            dict = {}
            dict['slurm_file'] = slurm_file_in
            dict['term_out'] = term_out

            run_dict.append(dict)

savepath = '/n/coxfs01/ybansal/sbatchfiles/noise_ens/run_details/' +  project_name + '_'+time.strftime("%Y-%m-%d_%H-%M-%S")+'.pkl'
with open(savepath, 'wb') as f:
    pkl.dump(run_dict, f)
