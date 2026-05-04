# coding: utf-8
__author__ = 'Ilya Kiselev (kiselecheck): https://github.com/kiselecheck'
__version__ = '1.0.1'

import os

import torch
import torch.multiprocessing as mp
from train import train_model
from utils.settings import cleanup_ddp, find_free_port, parse_args_train
import warnings

warnings.filterwarnings("ignore")


def train_model_single(rank: int, world_size: int, args=None):
    """
    Trains the model based on the provided arguments, including data preparation, optimizer setup,
    and loss calculation. The model is trained for multiple epochs with logging via wandb.

    Args:
        world_size:
        rank:
        args: Command-line arguments containing configuration paths, hyperparameters, and other settings.

    Returns:
        None
    """
    try:
        train_model(args, rank, world_size)
    finally:
        cleanup_ddp()


def train_model_ddp(args=None):
    args = parse_args_train(args)
    if not torch.cuda.is_available():
        raise RuntimeError("DDP training requires CUDA devices.")

    available_devices = torch.cuda.device_count()
    invalid_device_ids = [device_id for device_id in args.device_ids if device_id < 0 or device_id >= available_devices]
    if invalid_device_ids:
        raise ValueError(f"Invalid CUDA device ids {invalid_device_ids}; available device count is {available_devices}.")

    world_size = len(args.device_ids)
    if world_size < 1:
        raise ValueError("At least one CUDA device id is required for DDP training.")

    if "MASTER_PORT" not in os.environ:
        os.environ["MASTER_PORT"] = str(find_free_port())

    mp.spawn(train_model_single, args=(world_size, args), nprocs=world_size, join=True)


if __name__ == "__main__":
    train_model_ddp()
