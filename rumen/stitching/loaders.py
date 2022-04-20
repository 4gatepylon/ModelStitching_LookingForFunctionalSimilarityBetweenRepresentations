# Enables type annotations using enclosing classes
from __future__ import annotations
from cifar import (
    FFCV_NORMALIZE_TRANSFORM,
    FFCV_CIFAR_MEAN,
    FFCV_CIFAR_STD,
    NO_FFCV_NORMALIZE_TRANSFORM,
    NO_FFCV_CIFAR_MEAN,
    NO_FFCV_CIFAR_STD,
)
from typing import List, Any, Tuple
import os
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
import torch
import torchvision

from trainer import Hyperparams
from warnings import warn

FFCV_AVAIL = True
try:
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
except:
    FFCV_AVAIL = False
    warn("WARNING: FFCV is not available, getting ffcv loaders may not run")


class Loaders(object):
    NO_FFCV_FOLDER = "../../data/"

    @staticmethod
    def get_loaders_ffcv(args: Any) -> Tuple[DataLoader, DataLoader]:
        if FFCV_AVAIL == False:
            raise RuntimeError(
                "Need FFCV available for get_loaders_ffcv, try get_loaders_no_ffcv for a vanilla implementation")
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
        train_loader = Loader(f'./tmp/finetune_{args.dataset}_{args.fraction}_train_data.beton',
                              batch_size=args.bsz,
                              num_workers=args.num_workers,
                              order=OrderOption.RANDOM,
                              os_cache=True,
                              drop_last=True,
                              pipelines={
                                  'image': image_pipeline_train,
                                  'label': label_pipeline,
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
            torchvision.transforms.Normalize(FFCV_CIFAR_MEAN, FFCV_CIFAR_STD)
        ]

        test_loader = Loader(f'./tmp/{args.dataset}_test_data.beton',
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

    @staticmethod
    def get_loaders_no_ffcv(args: Any) -> Tuple[DataLoader, DataLoader]:
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
            transforms.Normalize(NO_FFCV_CIFAR_MEAN, NO_FFCV_CIFAR_STD)
        ])

        dataset1 = datasets.MNIST(
            Loaders.NO_FFCV_FOLDER, train=True, download=True, transform=transform)
        dataset2 = datasets.MNIST(
            Loaders.NO_FFCV_FOLDER, train=False, transform=transform)

        train_loader = DataLoader(dataset1, **train_kwargs)
        test_loader = DataLoader(dataset2, **test_kwargs)

        return train_loader, test_loader

    @staticmethod
    def get_loaders(args: Any) -> Tuple[DataLoader, DataLoader]:
        if args.use_ffcv:
            return Loaders.get_loaders_ffcv(args)
        else:
            return Loaders.get_loaders_no_ffcv(args)


class MockDataset(Dataset):
    """ Mocks dataset so that we can test other things (like loaders) """
    NUM_SAMPLES = 1000
    Xs = [torch.rand((3, 32, 32)) for _ in range(NUM_SAMPLES)]
    Ys = [torch.tensor(0) for _ in range(NUM_SAMPLES)]

    def __init__(self):
        self.X_Y = list(zip(MockDataset.Xs, MockDataset.Ys))

    def __len__(self):
        return len(self.X_Y)

    def __getitem__(self, idx):
        return self.X_Y[idx]


class MockDataLoader(DataLoader):
    """ Mocks a dataloader so that we can test training, freezing, etcetera """

    def __init__(self: MockDataLoader):
        pass

    @ staticmethod
    def mock(batch_size: int) -> MockDataLoader:
        assert batch_size == 1
        return DataLoader(
            dataset=MockDataset(),
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
        )

    @ staticmethod
    def mock_loaders(hyps: Hyperparams) -> Tuple[DataLoader, DataLoader]:
        return (
            MockDataLoader.mock(hyps.bsz),
            MockDataLoader.mock(hyps.bsz),
        )


if __name__ == "__main__":
    Loaders.get_loaders_no_ffcv(Hyperparams.forTesting())
    pass
