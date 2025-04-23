import logging
from copy import deepcopy
import numpy as np

from omegaconf import DictConfig
from hydra.utils import instantiate

import torch
from torch.utils.data import DataLoader

import wandb
from tqdm import tqdm

from utils.fabric import setup_fabric
from src.method.composer import Composer

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)

def get_dataloader(config, fabric):
    return fabric.setup_dataloaders(instantiate(config.dataset))

def run(config: DictConfig):
    log.info(f'Launching Fabric')
    fabric = setup_fabric(config)

    log.info(f'Building model')
    model = fabric.setup(instantiate(config.network))

    log.info(f'Setting up method')
    method = instantiate(config.method)(model)

    log.info(f'Setting up dataloaders')
    dataloader = get_dataloader(config, fabric)

    for batch_idx, (X,y) in enumerate(dataloader):
        X = fabric.to_device(X)
        y = fabric.to_device(y)

        output_bounds = method.forward(X,y)
        print(output_bounds)
        