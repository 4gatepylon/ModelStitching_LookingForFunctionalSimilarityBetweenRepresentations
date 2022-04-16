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

import torchvision
import torch
import torchvision

import os
from typing import List

from cifar import (
    NORMALIZE_TRANSFORM,
    CIFAR_MEAN,
    CIFAR_STD,
)


class Loaders(object):
    LABEL_PIPELINE: List[Operation] = [
        IntDecoder(),
        ToTensor(),
        ToDevice('cuda:0'),
        Squeeze(),
    ]

    def get_loaders_ffcv(args):
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
                                  'label': Loaders.LABEL_PIPELINE
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
                                 'label': Loaders.LABEL_PIPELINE
                             })
        return train_loader, test_loader
