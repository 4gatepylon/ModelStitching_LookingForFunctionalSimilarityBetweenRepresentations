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

import time
import math

# https://docs.python.org/3/library/typing.html#typing.ParamSpec
T = TypeVar('T')
P = ParamSpec('P')


class Hyperparams(object):
    """ Class that replaces `args` from the argument parser. Will have utility later. """

    def __init__(self):
        # FFCV Number of workers for loading
        self.fccv_num_workers: int = 1

        # Used by FFCV for train/test split
        self.fraction = 1.0

        # Training Hyperparams
        self.bsz = 256   # Batch Size
        self.lr = 0.01   # Learning Rate
        self.warmup = 10  # Warmup epochs
        self.epochs = 4  # Total epochs
        self.wd = 0.01   # Weight decay

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
                    # print(h.shape)
                    preds = h.argmax(dim=1)
                    # print(preds[0])
                    # print(labels[0])
                    total_correct = (preds == labels).sum().cpu().item()
                    # print(total_correct)
                    total_num += h.shape[0]

        return total_correct / total_num * 100.

    @staticmethod
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

    @staticmethod
    def train_loop(
        args: Any,
        model: nn.Module,
        train_loader: DataLoader,
        test_loader: DataLoader,
        parameters: Optional[List[torch.Tensor]] = None,
        epochs: Optional[int] = None,
    ) -> int:
        # None signifies do all parameters (we might finetune single layers an that will speed up training)
        if parameters is None:
            parameters = list(model.parameters())

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
            model.train()
            # epoch
            # NOTE that enumerate's start changes the starting index
            for it, (inputs, y) in enumerate(train_loader, start=(e - 1) * len(train_loader)):

                # adjust
                Trainer.adjust_learning_rate(epochs=epochs,
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
                    loss = F.cross_entropy(h, y)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

            print(f'\t\tepoch: {e} | time: {time.time() - start:.3f}')

        eval_acc = Trainer.evaluate(model, test_loader)

        return eval_acc
