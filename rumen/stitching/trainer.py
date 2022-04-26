# Enables type annotations using enclosing classes
from __future__ import annotations

import unittest

# Enables more interesting type annotations
from typing_extensions import (
    Concatenate,
    ParamSpec,
)
from typing import (
    Dict,
    NoReturn,
    Callable,
    Union,
    List,
    Any,
    Optional,
    Tuple,
    TypeVar,
)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import Dataset, DataLoader

from cifar import pclone, listeq

import time
import math

# https://docs.python.org/3/library/typing.html#typing.ParamSpec
T = TypeVar('T')
P = ParamSpec('P')

# In theory it's faster this way...
def adjust_learning_rate(
    epochs: int,
    warmup_epochs: int,
    base_lr: int,
    optimizer: Any,
    loader: DataLoader,
    step: int,
) -> NoReturn:
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


class Hyperparams(object):
    """ Class that replaces `args` from the argument parser. Will have utility later. """

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

        # In theory it's fast
        self.use_ffcv = True

        # options used for experiment details
        # two_way toggles whether to stitch the second model into the first
        # while control toggles whether to create control models
        # (True True -> 8 trainings; False True -> 4 trainings; False False -> 1 training)
        self.two_way = False
        # self.control = True
        self.control = False
        # Alternative is "cifar100"
        self.dataset = "cifar10"

    @staticmethod
    def forTesting():
        hyps = Hyperparams()
        hyps.lr = 1
        hyps.epochs = 1
        hyps.bsz = 1
        hyps.wd = 0.0
        hyps.warmup = 1
        hyps.fraction = 1.0
        hyps.ffcv_num_workers = 1
        hyps.two_way = False
        hyps.control = False
        return hyps


class Trainer(object):
    """ Just a class to encapsulate our training methods """

    def __init__(self: Trainer):
        pass

    @staticmethod
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

    @staticmethod
    def train_loop(
        args: Any,
        model: nn.Module,
        train_loader: DataLoader,
        test_loader: DataLoader,
        parameters: Optional[List[torch.Tensor]] = None,
        epochs: Optional[int] = None,
        verbose: bool = True,
    ) -> int:
        # None signifies do all parameters (we might finetune single layers an that will speed up training)
        if parameters is None:
            parameters = list(model.parameters())
        # print(parameters)

        optimizer = torch.optim.SGD(
            params=parameters,
            momentum=0.9,
            lr=args.lr * args.bsz / 256,
            weight_decay=args.wd
        )

        scaler = GradScaler()

        start = time.time()
        epochs = args.epochs if epochs is None else epochs
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

        eval_acc = Trainer.evaluate(model, test_loader)

        # Nothing should change here (otherwise maybe the weights changed when
        # we did not want them to)
        assert Trainer.evaluate(model, test_loader) == eval_acc

        return eval_acc


# Very similar to that one in loaders, but here we just need somewhere we can predict the optimum for
class MockDataset(Dataset):
    NUM_SAMPLES = 100
    Xs = [torch.rand(2)*10 for _ in range(NUM_SAMPLES)]
    Ys = [torch.tensor(0) for _ in range(NUM_SAMPLES)]

    def __init__(self):
        self.X_Y = list(zip(MockDataset.Xs, MockDataset.Ys))

    def __len__(self):
        return len(self.X_Y)

    def __getitem__(self, idx):
        return self.X_Y[idx]

# TODO this does not seem to properly learn
class TrainerTester(unittest.TestCase):
    def test_train_loop(self):
        # Get hyperparameters with bsz = 1
        hyps = Hyperparams.forTesting()
        hyps.epochs = 100

        # Should be converging to [1, 0]
        model = nn.Linear(2, 1, bias = False)
        original_params = pclone(model)

        test_loader = DataLoader(MockDataset(), batch_size=hyps.bsz)
        train_loader = DataLoader(MockDataset(), batch_size=hyps.bsz)
        acc = Trainer.train_loop(hyps, model, train_loader, test_loader, verbose=False)
        self.assertTrue(acc > 0.0)

        optimum_weights = torch.tensor([1, 0]).float()
        original_dist = torch.dist(original_params[0], optimum_weights, p=2)
        new_dist = torch.dist(model.weight, optimum_weights, p=2)

        # Ascertain that the weights change and that they get better
        self.assertFalse(listeq(original_params, list(model.parameters())))
        self.assertTrue(new_dist < original_dist)

if __name__ == "__main__":
    unittest.main(verbosity=2)
