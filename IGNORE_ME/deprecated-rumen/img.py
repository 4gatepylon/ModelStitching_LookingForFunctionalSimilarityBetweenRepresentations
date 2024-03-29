from ffcv.writer import DatasetWriter
from ffcv.transforms.common import Squeeze
from ffcv.transforms import RandomHorizontalFlip, Cutout, \
    RandomTranslate, Convert, ToDevice, ToTensor, ToTorchImage
from ffcv.pipeline.operation import Operation
from ffcv.loader import Loader, OrderOption
from ffcv.fields.decoders import IntDecoder, SimpleRGBImageDecoder, RandomResizedCropRGBImageDecoder
from ffcv.fields import IntField, RGBImageField

import torchvision
import torchvision.transforms.functional as TF
import torch
import torchvision
from torch.cuda.amp import autocast

import cv2
import os
from typing import List

CIFAR_MEAN = [125.307, 122.961, 113.8575]
CIFAR_STD = [51.5865, 50.847, 51.255]
INV_CIFAR_MEAN = [-ans for ans in CIFAR_MEAN]
INV_CIFAR_STD = [1.0/ans for ans in CIFAR_STD]

NORMALIZE_TRANSFORM = torchvision.transforms.Normalize(CIFAR_MEAN, CIFAR_STD)
INV_NORMALIZE_TRANSFORM = torchvision.transforms.Compose([
    torchvision.transforms.Normalize(mean=[0, 0, 0], std=INV_CIFAR_STD),
    torchvision.transforms.Normalize(mean=INV_CIFAR_MEAN, std=[1, 1, 1])
])

# Dataloading using FFCV (is much faster)

label_pipeline: List[Operation] = [
    IntDecoder(), ToTensor(), ToDevice('cuda:0'), Squeeze()]


def get_loaders(args):
    num_of_points = 50000
    split = [int(num_of_points * args.fraction),
             int(num_of_points * (1 - args.fraction))]

    dataset_class = torchvision.datasets.CIFAR10 if (
        args.dataset == 'cifar10') else torchvision.datasets.CIFAR100
    if not os.path.exists(f'tmp/finetune_{args.dataset}_{args.fraction}_train_data.beton'):
        train_data = dataset_class(
            '../Data', train=True, download=True
        )
        train_data = torch.utils.data.random_split(train_data, split)[0]
        train_writer = DatasetWriter(f'tmp/finetune_{args.dataset}_{args.fraction}_train_data.beton', {
            'image': RGBImageField(),
            'label': IntField()
        })
        train_writer.from_indexed_dataset(train_data)

    image_pipeline_train: List[Operation] = [
        SimpleRGBImageDecoder(),
        RandomHorizontalFlip(),
        RandomTranslate(padding=2, fill=tuple(map(int, CIFAR_MEAN))),
        Cutout(4, tuple(map(int, CIFAR_MEAN))),
        ToTensor(),
        ToDevice('cuda:0', non_blocking=True),
        ToTorchImage(),
        Convert(torch.float16),
        NORMALIZE_TRANSFORM
    ]
    train_loader = Loader(f'./tmp/finetune_{args.dataset}_{args.fraction}_train_data.beton',
                          batch_size=args.bsz,
                          num_workers=args.num_workers,
                          order=OrderOption.RANDOM,
                          os_cache=True,
                          drop_last=True,
                          pipelines={
                              'image': image_pipeline_train,
                              'label': label_pipeline
                          })

    if not os.path.exists(f'tmp/{args.dataset}_test_data.beton'):
        test_data = dataset_class(
            '../Data', train=False, download=True
        )

        test_writer = DatasetWriter(f'tmp/{args.dataset}_test_data.beton', {
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
        torchvision.transforms.Normalize(CIFAR_MEAN, CIFAR_STD)
    ]

    test_loader = Loader(f'./tmp/{args.dataset}_test_data.beton',
                         batch_size=2048,
                         num_workers=args.num_workers,
                         order=OrderOption.SEQUENTIAL,
                         os_cache=True,
                         drop_last=False,
                         pipelines={
                             'image': image_pipeline_test,
                             'label': label_pipeline
                         })
    return train_loader, test_loader


def tensor_normalized2rgb(x: torch.Tensor):
    f = x.float()
    y = INV_NORMALIZE_TRANSFORM(f)
    return y


def tensor_normalized2pil_img(x: torch.Tensor):
    rgb_tensor = tensor_normalized2rgb(x)
    return TF.to_pil_image(rgb_tensor)


def tensor_normalized2nd_array(x: torch.Tensor):
    rgb_tensor = tensor_normalized2rgb(x)
    return rgb_tensor.numpy()


# This will also be helpful!
# https://pytorch.org/vision/master/auto_examples/plot_visualization_utils.html
# https://medium.com/analytics-vidhya/read-image-using-cv2-imread-opencv-python-idiot-developer-4b7401e76c20
# https://www.tutorialkart.com/opencv/python/opencv-python-save-image-example/
IMG_FILE = "human_interpretable_img.png"


def show_tensor_normalized(x: torch.Tensor, file=IMG_FILE):
    nd_array = tensor_normalized2nd_array(x)
    cv2.imwrite(file, nd_array)
    pass


IMG_FILE_PREFIX = ""


def show_n_tensors_normalized(num, args):
    i = 0
    train_loader, test_loader = get_loaders(args)
    # TODO iterate through the first few images of the first batch or if necessary overflow batches
    # (look at cifar_supervised) since there is a function there that does it
    pass

# TODO should be used above ^


def label_before(label, model_blocks):
    if label == "conv1":
        # raise Exception("Cannot get label before conv1")
        return "input"
    if label == "fc":
        return (4, model_blocks[4-1] - 1)
    layer, block = label
    if block > 0:
        return (layer, block - 1)
    elif layer > 1:
        return (layer - 1, model_blocks[layer - 1 - 1] - 1)
    else:
        return "conv1"

# mean squared differnece over all training examples
# of the vanilla stitch and the stitched model


def mean2_model_diff(stitch, sender, reciever, snd_label, rcv_label, model_blocks, train_loader, stitch2=None):
    # out-vent from the
    # NOTE we use train_loader but it really doesn't matter
    num_images = len(train_loader)
    total = 0.0
    label_before_rcv_label = label_before(rcv_label, model_blocks)
    for x, _ in train_loader:
        with autocast():
            # print(f"x shape {x.size()}")
            sent = sender(x, vent=snd_label, into=False)
            # print(f"sent shape {sent.size()}")
            # print(f"label before {label_before_rcv_label}")
            expected = reciever(x, vent=label_before_rcv_label, into=False,
                                apply_post=True) if stitch2 is None else stitch2(sent)
            # print(f"expected shape {expected.size()}")

            gotten = stitch(sent)
            # print(f"gotten shape {gotten.size()}")
            # Average mean squared error over the batch and pixels
            diff = (gotten - expected).pow(2).mean()
            total += diff.cpu().item()
        pass
    # Average over the number of images
    total /= num_images
    return total


def get_n_inputs(n, loader):
    k = 0
    for x, _ in loader:
        if k > n:
            break
        batch_size, _, _, _ = x.size()
        # print(f"batch size {batch_size}")
        for i in range(min(batch_size, n - k)):
            # Output as a 4D tensor so that the network can take this as input
            y = x[i, :, :, :].flatten(end_dim=0).unflatten(0, (1, -1))
            # print(y.size())
            yield y
        k += batch_size

# def save_random_image_pairs(st, sender, snd_label, num_pairs, foldername_images, train_loader):
#     original_tensors = list(get_n_inputs(num_pairs, train_loader))
#     for i in range(num_pairs):
#         # Pick the filenames
#         original_filename = os.path.join(
#             foldername_images, f"original_{i}.png")
#         generated_filename = os.path.join(
#             foldername_images, f"generated_{i}.png")

#         with autocast():
#             original_tensor = original_tensors[i]
#             print(f"\tog tensor shape is {original_tensor.size()}")
#             generated_tensor_pre = sender(
#                 original_tensor, vent=snd_label, into=False)
#             generated_tensor = st(generated_tensor_pre)

#         # Save the images
#         original_tensor_flat = original_tensor.flatten(end_dim=0)
#         generated_tensor_flat = generated_tensor.flatten(end_dim=0)
#         original_np = original_tensor_flat.cpu().numpy()
#         generated_np = generated_tensor_flat.cpu().numpy()
#         cv2.imwrite(original_np, original_filename)
#         cv2.imwrite(generated_np, generated_filename)
#         # TDOO


if __name__ == '__main__':
    # TODO
    show_n_tensors_normalized(5, None)
    pass
