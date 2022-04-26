# A Massive script to try and do the entire stitching experiment (in its simplest possible form)
# without any external context.

from ffcv.writer import (
    DatasetWriter,
)
from ffcv.transforms.common import (
    Squeeze,
)
from ffcv.transforms import (
    RandomHorizontalFlip,
    Cutout,
    RandomTranslate,
    Convert,
    ToDevice,
    ToTensor,
    ToTorchImage,
)
from ffcv.pipeline.operation import (
    Operation,
)
from ffcv.loader import (
    Loader,
    OrderOption,
)
from ffcv.fields.decoders import (
    IntDecoder,
    SimpleRGBImageDecoder,
    RandomResizedCropRGBImageDecoder,
)
from ffcv.fields import (
    IntField,
    RGBImageField,
)

import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.cuda.amp import GradScaler, autocast

from mega_resnet import make_resnet
from mega_resnet import make_stitched_resnet

# Used in the experiment to create the tables
NUM_LAYERS = 6

# NOTE same as the original because it's the same tree height away
RESNETS_FOLDER = "../../pretrained_resnets/"
SIMS_FOLDER = "../../sims/"
HEATMAPS_FOLDER = "../../heatmaps/"

# Loaders folders
FFCV_FOLDER = "../../data_ffcv/"
MISC_FFCV_FOLDER = "../tmp/"

FFCV_CIFAR_MEAN = [125.307, 122.961, 113.8575]
FFCV_CIFAR_STD = [51.5865, 50.847, 51.255]
FFCV_NORMALIZE_TRANSFORM = torchvision.transforms.Normalize(
    FFCV_CIFAR_MEAN, FFCV_CIFAR_STD)

def get_loaders(args):
    if FFCV_AVAIL == False:
        raise RuntimeError(
            "Need FFCV available for get_loaders_ffcv, try get_loaders_no_ffcv for a vanilla implementation")
    num_of_points = 50000
    split = [int(num_of_points * args.fraction),
                int(num_of_points * (1 - args.fraction))]

    dataset_class = torchvision.datasets.CIFAR10 if (
        args.dataset == 'cifar10') else torchvision.datasets.CIFAR100
    finetune_file = os.path.join(
        MISC_FFCV_FOLDER, f"finetune_{args.dataset}_{args.fraction}_train_data.beton")
    if not os.path.exists(finetune_file):
        train_data = dataset_class(
            FFCV_FOLDER, train=True, download=True
        )
        train_data = torch.utils.data.random_split(train_data, split)[0]
        train_writer = DatasetWriter(finetune_file, {
            'image': RGBImageField(),
            'label': IntField()
        })
        train_writer.from_indexed_dataset(train_data)

    label_pipeline: List[Operation] = [
        IntDecoder(),
        ToTensor(),
        ToDevice('cuda:0'),
        Squeeze(),
    ]

    image_pipeline_train: List[Operation] = [
        SimpleRGBImageDecoder(),
        RandomHorizontalFlip(),
        RandomTranslate(padding=2, fill=tuple(map(int, FFCV_CIFAR_MEAN))),
        Cutout(4, tuple(map(int, FFCV_CIFAR_MEAN))),
        ToTensor(),
        ToDevice('cuda:0', non_blocking=True),
        ToTorchImage(),
        Convert(torch.float16),
        FFCV_NORMALIZE_TRANSFORM
    ]
    train_loader = Loader(finetune_file,
                            batch_size=args.bsz,
                            num_workers=args.num_workers,
                            order=OrderOption.RANDOM,
                            os_cache=True,
                            drop_last=True,
                            pipelines={
                                'image': image_pipeline_train,
                                'label': label_pipeline,
                            })

    test_data_file = os.path.join(
        MISC_FFCV_FOLDER, f"{args.dataset}_test_data.beton")
    if not os.path.exists(test_data_file):
        test_data = dataset_class(
            FFCV_FOLDER, train=False, download=True
        )

        test_writer = DatasetWriter(test_data_file, {
            'image': RGBImageField(),
            'label': IntField()
        })
        test_writer.from_indexed_dataset(test_data)

    image_pipeline_test: List[Operation] = [
        SimpleRGBImageDecoder(),
        ToTensor(),
        ToDevice('cuda:0', non_blocking=True),
        ToTorchImage(),
        Convert(torch.float16),
        torchvision.transforms.Normalize(FFCV_CIFAR_MEAN, FFCV_CIFAR_STD)
    ]

    test_loader = Loader(test_data_file,
                            batch_size=2048,
                            num_workers=args.num_workers,
                            order=OrderOption.SEQUENTIAL,
                            os_cache=True,
                            drop_last=False,
                            pipelines={
                                'image': image_pipeline_test,
                                'label': label_pipeline,
                            })
    return train_loader, test_loader

def adjust_learning_rate(
    epochs,
    warmup_epochs,
    base_lr,
    optimizer,
    loader,
    step,
):
    max_steps = epochs * len(loader)
    warmup_steps = warmup_epochs * len(loader)
    if step < warmup_steps:
        lr = base_lr * step / warmup_steps
    else:
        step -= warmup_steps
        max_steps -= warmup_steps
        q = 0.5 * (1 + math.cos(math.pi * step / max_steps))
        end_lr = 0
        lr = base_lr * q + end_lr * (1 - q)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def evaluate(model, test_loader):
    # NOTE used to be for layer in model
    model.eval()

    for _, (images, labels) in enumerate(test_loader):
        total_correct, total_num = 0., 0.

        with torch.no_grad():
            with autocast():
                h = images
                h = model(h)
                preds = h.argmax(dim=1)
                total_correct = (preds == labels).sum().cpu().item()
                total_num += h.shape[0]

    return total_correct / total_num

def train_loop(
    args,
    model,
    train_loader,
    test_loader,
    verbose = True,
):
    parameters = list(model.parameters())

    optimizer = torch.optim.SGD(
        params=parameters,
        momentum=0.9,
        lr=args.lr * args.bsz / 256,
        weight_decay=args.wd
    )

    scaler = GradScaler()

    start = time.time()
    epochs = args.epochs
    for e in range(1, epochs + 1):
        if verbose:
            print(f"\t\t starting on epoch {e} for {len(train_loader)} iterations")
        model.train()
        # epoch
        # NOTE that enumerate's start changes the starting index
        for it, (inputs, y) in enumerate(train_loader, start=(e - 1) * len(train_loader)):
            # TODO not sure why it's so slow sometimes, but it seems to need to "Warm up"
            # ... I've never seen this before ngl
            # print(f"\t\t\titeration {it}")
            # adjust
            adjust_learning_rate(epochs=epochs,
                                    warmup_epochs=args.warmup,
                                    base_lr=args.lr * args.bsz / 256,
                                    optimizer=optimizer,
                                    loader=train_loader,
                                    step=it)
            # zero grad (should we set to none?)
            optimizer.zero_grad(set_to_none=True)

            with autocast():
                h = inputs
                h = model(h)
                #print(h)
                #print(y)
                # TODO modularize this out to enable sim training
                loss = F.cross_entropy(h, y)
                #print(loss)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        if verbose:
            print(f'\t\tepoch: {e} | time: {time.time() - start:.3f}')

    eval_acc = evaluate(model, test_loader)

    return eval_acc

# Tells you the shape that's coming INTO a layer
# Which is either a 3-tuple for tensors or a single scalar for the width
# of a linear layer
def get_prev_shape(label):
    if label == "input":
        raise Exception("input layer has no previous shape")
    if label == "conv1":
        return (3, 32, 32)
    if label == "fc":
        return (512, 4, 4)
    if label == "output":
        return 10
    if type(label) == tuple:
        blockset, _ = label
        if blockset == 1 or blockset == 2:
            return (64, 32, 32)
        if blockset == 3:
            return (128, 16, 16)
        if blockset == 4:
            return (256, 8, 8)
        else:
            raise Exception(f"unknown tuple label {label} in get_prev_shape")
    else:
        raise Exception(f"unknown label: {label} in get_prev_shape")

def make_stitch(send_label, recv_label):
        send_shape = get_prev_shape(send_label)
        recv_shape = get_prev_shape(recv_label)
        assert type(send_shape) == tuple
        assert len(send_shape) == 3
        # assert type(recv_shape) == tuple or t
        if type(recv_shape) == int:
            return nn.Sequential(
            nn.Flatten(),
            nn.Linear(
                send_shape[0] * send_shape[1] * send_shape[2],
                recv_shape,
            ),
        )
        else:
            send_depth, send_height, send_width = send_shape
            recv_depth, recv_height, recv_width = recv_shape
            if recv_height <= send_height:
                ratio = send_height // recv_height
                return nn.Conv2d(
                    send_depth,
                    recv_depth,
                    ratio,
                    stride=ratio,
                    bias=StitchGenerator.USE_BIAS,
                )
            else:
                ratio = recv_height // send_height
                return nn.Sequential(
                    nn.Upsample(
                        scale_factor=ratio,
                        mode='nearest',
                    ),
                    nn.Conv2d(
                        send_depth,
                        recv_depth,
                        1,
                        stride=1,
                        bias=StitchGenerator.USE_BIAS,
                    ),
                )

def stitchtrain(args):
        numbers1 = [1, 1, 1, 1]
        _file = os.path.join(RESNETS_FOLDER, "resnet_1111")

        # TODO port this over
        print("Loading Model and moving to Cuda")
        model = make_resnet(
            name,
            BasicBlock,
            number,
            False,
            False,
            num_classes=10,
        ) for name, number in zip(names, numbers)
        model = model.cuda()
        model.load_state_dict(torch.load(_file))
        
        print("Getting loaders for FFCV")
        train_loader, test_loader = get_loaders(args)

        acc = Trainer.evaluate(model, test_loader)
        print("Model Original Accuracy: {}".format(acc))

        # Everyone recieves, but not everyone sends
        # We do it this way for maximal simplicity
        print("Creating Tables")
        layerlabels = [
            [("conv1", "conv1"), ("conv1", (1, 0)), ("conv1", (2, 0)), ("conv1", (3, 0)), ("conv1", (4, 0)), ("conv1", "fc")],
            [((1, 0), "conv1"), ((1, 0), (1, 0)), ((1, 0), (2, 0)), ((1, 0), (3, 0)), ((1, 0), (4, 0)), ((1, 0), "fc")],
            [((2, 0), "conv1"), ((2, 0), (1, 0)), ((2, 0), (2, 0)), ((2, 0), (3, 0)), ((2, 0), (4, 0)), ((2, 0), "fc")],
            [((3, 0), "conv1"), ((3, 0), (1, 0)), ((3, 0), (2, 0)), ((3, 0), (3, 0)), ((3, 0), (4, 0)), ((3, 0), "fc")],
            [((4, 0), "conv1"), ((4, 0), (1, 0)), ((4, 0), (2, 0)), ((4, 0), (3, 0)), ((4, 0), (4, 0)), ((4, 0), "fc")],
        ]
        stitches = [
           [make_stitch(label1, label2) for (label1, label2) in row] for row in layerlabels
        ]
        sims = [
            # Use -1.0 as a signal that this failed to train (since that's impossible)
            [-1.0 for _ in range(NUM_LAYERS)] \
            for _ in range(NUM_LAYERS - 1)
        ]

        # Make sure all the lengths are correct
        assert len(layerlabels) == NUM_LAYERS - 1
        assert len(stitches) == NUM_LAYERS - 1
        assert len(sims) == NUM_LAYERS - 1
        assert max((len(l) for l in layerlabels)) == NUM_LAYERS
        assert max((len(l) for l in stitches)) == NUM_LAYERS
        assert max((len(l) for l in sims)) == NUM_LAYERS
        assert min((len(l) for l in layerlabels)) == NUM_LAYERS
        assert min((len(l) for l in stitches)) == NUM_LAYERS
        assert min((len(l) for l in sims)) == NUM_LAYERS

        print("Training")
        for i in range(NUM_LAYERS):
            for j in range(NUM_LAYERS - 1):
                send_label, recv_label = layerlabels[j][i]
                stitch = stitches[i][j]
                stitched_resnet = make_stitched_resnet(model, stitch, send_label, recv_label)
                acc = train_loop(args, model, train_loader, test_loader)
                sims[i][j] = acc

        print("Saving similarities")
        if not os.path.exists(SIMS_FOLDER):
            os.mkdir(SIMS_FOLDER)
        if not os.path.exists(HEATMAPS_FOLDER):
            os.mkdir(HEATMAPS_FOLDER)
        torch.save(torch.tensor(sims), sim_path)
        Visualizer.matrix_heatmap(sim_path, heat_path)

class Args:
    def __init__(self):
        # FFCV Number of workers for loading
        self.fccv_num_workers: int = 1
        self.num_workers = 1

        # Used by FFCV for train/test split
        self.fraction = 1.0

        # Training Hyperparams
        self.bsz = 256   # Batch Size
        self.lr = 0.01   # Learning Rate
        self.warmup = 10  # Warmup epochs
        self.epochs = 4  # Total epochs
        self.wd = 0.01   # Weight decay
        self.dataset = "cifar10"

if __name__ == "__main__":
    args = Args()
    stitchtrain(args)