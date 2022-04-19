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
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

import os
from typing import List, Any, Tuple

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

    NO_FFCV_FOLDER = "../../data/"

    @staticmethod
    def get_loaders_ffcv(args: Any) -> Tuple[DataLoader, DataLoader]:
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

    @staticmethod
    def get_dataloaders_not_ffcv(args: Any) -> Tuple[DataLoader, DataLoader]:
        use_cuda = torch.cuda.is_available()
        train_batch_size = args.bsz
        test_batch_size = 2048

        device = torch.device("cuda" if use_cuda else "cpu")

        train_kwargs = {'batch_size': train_batch_size}
        test_kwargs = {'batch_size': test_batch_size}
        if use_cuda:
            cuda_kwargs = {'num_workers': 1,
                           'pin_memory': True}
            train_kwargs.update(cuda_kwargs)
            test_kwargs.update(cuda_kwargs)
        train_kwargs['shuffle'] = True
        test_kwargs['shuffle'] = True

        transform = transforms.Compose([
            transforms.ToTensor(),
            # Unclear which should be used?
            # transforms.Normalize(CIFAR_MEAN, CIFAR_STD)
            transforms.Normalize((0.1307,), (0.3081,))
        ])

        dataset1 = datasets.MNIST(
            Loaders.NO_FFCV_FOLDER, train=True, download=False, transform=transform)
        dataset2 = datasets.MNIST(
            Loaders.NO_FFCV_FOLDER, train=False, transform=transform)

        train_loader = DataLoader(dataset1, **train_kwargs)
        test_loader = DataLoader(dataset2, **test_kwargs)

        return train_loader, test_loader
