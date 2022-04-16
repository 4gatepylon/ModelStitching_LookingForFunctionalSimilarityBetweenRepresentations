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
    def __init__(self):
        self.seed: Optional[int] = None

        # FFCV Number of workers for loading
        self.fccv_num_workers: int = 1

        # Used by FFCV for train/test split
        self.fraction = 1.0

        raise NotImplementedError


class ExperimentHyperparams(Hyperparams):
    def __init__(self: ExperimentHyperparams, trainer_hyperparams: TrainerHyperparams) -> NoReturn:
        super().__init__()

        self.trainer_hyperparams = trainer_hyperparams

        # TODO
        raise NotImplementedError


class TrainerHyperparams(Hyperparams):
    def __init__(self: TrainerHyperparams):
        super().__init__()

        self.bsz = 256   # Batch Size
        self.lr = 0.01   # Learning Rate
        self.warmup = 10  # Warmup epochs
        self.epochs = 1  # Total epochs
        self.wd = 0.01   # Weight decay

        raise NotImplementedError


class Trainer(object):
    def __init__(self: Trainer, hyperparams: TrainerHyperparams):
        # TODO
        pass

    def adjust_learning_rate(self: Trainer, epochs: int, warmup_epochs, base_lr, optimizer, loader, step):
        epochs: int = self.epochs
        warmup_epochs: int = self.warmup
        base_lr: int = self.lr
        optimizer: torch.optim.Optimizer = self.optimizer
        loader: DataLoader = self.loader
        step: int = self.step

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

        return lr

    def train_loop(
        self: Trainer,
        model: nn.Module,
        train_loader: DataLoader,
        test_loader: DataLoader,
        parameters: Optional[List[torch.Tensor]] = None,
        epochs: Optional[int] = None,
    ):
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
                adjust_learning_rate(epochs=epochs,
                                     warmup_epochs=args.warmup,
                                     base_lr=args.lr * args.bsz / 256,
                                     optimizer=optimizer,
                                     loader=train_loader,
                                     step=it)
                # zero grad
                optimizer.zero_grad(set_to_none=True)

                with autocast():
                    h = inputs
                    h = model(h)
                    loss = F.cross_entropy(h, y)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

            print(f'\t\tepoch: {e} | time: {time.time() - start:.3f}')

        eval_acc = evaluate(model, test_loader)

        return eval_acc, model
