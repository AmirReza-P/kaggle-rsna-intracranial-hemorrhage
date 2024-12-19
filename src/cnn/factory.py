import copy

import torch
from torch import nn
from torch.utils.data import DataLoader
import torch.optim
from torch.optim import lr_scheduler
import albumentations as A
from albumentations.pytorch import ToTensor
import pretrainedmodels

from .dataset.custom_dataset import CustomDataset
from .transforms.transforms import RandomResizedCrop, RandomDicomNoise
from .utils.logger import log


def get_loss(cfg):
    #loss = getattr(nn, cfg.loss.name)(**cfg.loss.params)
    loss = getattr(nn, cfg.loss.name)(weight=torch.FloatTensor([2,1,1,1,1,1]).cuda(), **cfg.loss.params)
    log('loss: %s' % cfg.loss.name)
    return loss


def get_dataloader(cfg, folds=None):
    dataset = CustomDataset(cfg, folds)
    log('use default(random) sampler')
    loader = DataLoader(dataset, **cfg.loader)
    return loader


def get_transforms(cfg):
    def get_object(transform):
        if hasattr(A, transform.name):
            return getattr(A, transform.name)
        else:
            return eval(transform.name)
    transforms = [get_object(transform)(**transform.params) for transform in cfg.transforms]
    return A.Compose(transforms)


import os
import torch

def get_model(cfg):
    log(f'model: {cfg.model.name}')
    log(f'pretrained: {cfg.model.pretrained}')

    local_model_path_50 = '/kaggle/input/your-dataset/se_resnext50_32x4d-a260b3a4.pth'
    local_model_path_101 = '/kaggle/input/your-dataset/se_resnext101_32x4d-3b2fe3d8.pth'

    if cfg.model.name == 'se_resnext50_32x4d' and os.path.exists(local_model_path_50):
        log('Loading se_resnext50 from local cache')
        model = pretrainedmodels.__dict__['se_resnext50_32x4d'](num_classes=1000, pretrained=None)
        model.load_state_dict(torch.load(local_model_path_50))
    elif cfg.model.name == 'se_resnext101_32x4d' and os.path.exists(local_model_path_101):
        log('Loading se_resnext101 from local cache')
        model = pretrainedmodels.__dict__['se_resnext101_32x4d'](num_classes=1000, pretrained=None)
        model.load_state_dict(torch.load(local_model_path_101))
    else:
        try:
            model_func = pretrainedmodels.__dict__[cfg.model.name]
        except KeyError:
            model_func = eval(cfg.model.name)

        model = model_func(num_classes=1000, pretrained=cfg.model.pretrained)

    # Custom last layers
    model.avg_pool = nn.AdaptiveAvgPool2d(1)
    model.last_linear = nn.Linear(
        model.last_linear.in_features,
        cfg.model.n_output,
    )
    return model


def get_optim(cfg, parameters):
    optim = getattr(torch.optim, cfg.optim.name)(parameters, **cfg.optim.params)
    log(f'optim: {cfg.optim.name}')
    return optim


def get_scheduler(cfg, optim, last_epoch):
    if cfg.scheduler.name == 'ReduceLROnPlateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(
            optim,
            **cfg.scheduler.params,
        )
        scheduler.last_epoch = last_epoch
    else:
        scheduler = getattr(lr_scheduler, cfg.scheduler.name)(
            optim,
            last_epoch=last_epoch,
            **cfg.scheduler.params,
        )
    log(f'last_epoch: {last_epoch}')
    return scheduler

