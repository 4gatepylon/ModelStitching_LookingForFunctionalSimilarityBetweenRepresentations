import torch, wandb
import torchvision.transforms as transforms
import torchvision.datasets as dsets
from torch.utils.data import Dataset, DataLoader
import torch.distributed as dist
import torch.multiprocessing as mp
import os
import time
import numpy as np
import sys
from torchvision.models.resnet import resnet50


class Config(object):
    num_workers = 4

def main_worker(gpu, cfg):
    print('Entered main_worker')
    
    dist.init_process_group(backend="nccl", init_method=cfg.init_method, world_size=cfg.num_gpus, rank=gpu)
    print('Initialized process')
    
    torch.cuda.set_device(gpu)
    print('Set device')
    
    print(cfg.batch_size, cfg.num_gpus, cfg.num_workers, gpu)

    data_dir = '/n/holystore01/LABS/barak_lab/Everyone/datasets/imagenet256'
    mean=[0.485, 0.456, 0.406]
    std=[0.229, 0.224, 0.225]

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

    train_dataset = dsets.ImageFolder(os.path.join(data_dir, 'train'), transform=transform_train)

    per_process_batch_size = cfg.batch_size // cfg.num_gpus + int(gpu < cfg.batch_size % cfg.num_gpus)
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    tr_data_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=per_process_batch_size, 
            num_workers=cfg.num_workers, pin_memory=True, sampler=train_sampler,
            drop_last=True)

    if gpu==0:
        print('Dataset loaded...')
        wandb.init(project=cfg.wandb_project_name, entity='yaminibansal')

    model = resnet50(pretrained=True).cuda()

    if gpu==0:
        print('Model loaded...')

    tic_first = time.time()
    tic = time.time()

    for (i, X) in enumerate(tr_data_loader):
        x = X[0].cuda()
        preds = model(x)
        
        if gpu==0:
            timeElapsed = time.time()-tic
            wandb.log({'Time for iter' : timeElapsed})
        tic = time.time()

    print(f'Time for 1 epoch = {time.time() - tic_first}')

## Distributed
if __name__ == '__main__':
    '''
    sys.arg[1] = number of GPUs
    sys.arg[2] = number of workers per GPU
    sys.arg[3] = batch size
    '''
    print('Started!')
    distributed = True
    cfg = Config()
    cfg.num_gpus = int(sys.argv[1])
    cfg.num_workers = int(sys.argv[2])
    cfg.batch_size = int(sys.argv[3])
    cfg.init_method = f'tcp://127.0.0.1:{np.random.randint(20000, 30000)}'
    cfg.wandb_project_name = 'profile-distributed-training'

    with open('/n/coxfs01/ybansal/settings/wandb_api_key.txt', 'r') as f:
        key = f.read().split('\n')[0]
    os.environ['WANDB_API_KEY'] = key    
    
    print('Trying to spawn...')
    mp.spawn(main_worker, nprocs=cfg.num_gpus, args=(cfg,))
