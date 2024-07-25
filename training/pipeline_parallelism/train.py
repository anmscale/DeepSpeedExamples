#!/usr/bin/env python3

import os
import argparse

import torch
import torch.distributed as dist

import torchvision
import torchvision.transforms as transforms
# from torchvision.models import AlexNet
from models import MLP
import pathlib
from typing import Tuple, List
import logging
import loguru
import re
import torch.optim as optim
import time
import deepspeed
from deepspeed.pipe import PipelineModule
from deepspeed.utils import RepeatingLoader

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

logger = loguru.logger

def log_dist(message: str,
             ranks: List[int] = [],
             level: int = logging.INFO) -> None:
    """Log messages for specified ranks only"""
    my_rank = int(os.environ.get("RANK", "0"))
    if my_rank in ranks:
        if level == logging.INFO:
            logger.info(f'[Rank {my_rank}] {message}')
        if level == logging.ERROR:
            logger.error(f'[Rank {my_rank}] {message}')
        if level == logging.DEBUG:
            logger.debug(f'[Rank {my_rank}] {message}')

def is_rank_0() -> bool:
    return int(os.environ.get("RANK", "0")) == 0



def cifar_trainset(local_rank, dl_path='/tmp/cifar10-data'):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.Lambda(lambda x: x.view(-1))  # Flatten the tensor
    ])

    # Ensure only one rank downloads.
    # Note: if the download path is not on a shared filesytem, remove the semaphore
    # and switch to args.local_rank
    dist.barrier()
    if local_rank != 0:
        dist.barrier()
    trainset = torchvision.datasets.CIFAR10(root=dl_path,
                                            train=True,
                                            download=True,
                                            transform=transform)
    if local_rank == 0:
        dist.barrier()
    return trainset


def get_args():
    parser = argparse.ArgumentParser(description='CIFAR')
    parser.add_argument('--local_rank',
                        type=int,
                        default=-1,
                        help='local rank passed from distributed launcher')
    parser.add_argument('-s',
                        '--steps',
                        type=int,
                        default=100,
                        help='quit after this many steps')
    parser.add_argument('-p',
                        '--pipeline-parallel-size',
                        type=int,
                        default=2,
                        help='pipeline parallelism')
    parser.add_argument('--backend',
                        type=str,
                        default='nccl',
                        help='distributed backend')
    parser.add_argument('--seed', type=int, default=123, help='PRNG seed')
    parser.add_argument('--save_dir',
                        type=str,
                        default='/mnt/local_storage/mlp_benchmark/deepspeed/',
                        help='model checkpoint directory')
    parser = deepspeed.add_config_arguments(parser)
    args = parser.parse_args()
    return args


def train_base(args):
    # set_seed(args.seed)

    net = MLP(num_classes=10)

    trainset = cifar_trainset(args.local_rank)

    engine, _, dataloader, __ = deepspeed.initialize(
        args=args,
        model=net,
        model_parameters=[p for p in net.parameters() if p.requires_grad],
        training_data=trainset)

    dataloader = RepeatingLoader(dataloader)
    data_iter = iter(dataloader)

    rank = dist.get_rank()
    gas = engine.gradient_accumulation_steps()

    criterion = torch.nn.CrossEntropyLoss()

    total_steps = args.steps * engine.gradient_accumulation_steps()
    step = 0
    for micro_step in range(total_steps):
        batch = next(data_iter)
        inputs = batch[0].to(engine.device)
        labels = batch[1].to(engine.device)
        
        outputs = engine(inputs)
        loss = criterion(outputs, labels)
        engine.backward(loss)
        engine.step()

        if micro_step % engine.gradient_accumulation_steps() == 0:
            step += 1
            if rank == 0 and (step % 10 == 0):
                print(f'step: {step:3d} / {args.steps:3d} loss: {loss}')



def join_layers_convnet(vision_model):
    layers = [
        *vision_model.features,
        vision_model.avgpool,
        lambda x: torch.flatten(x, 1),
        *vision_model.classifier,
    ]
    return layers

def join_layers_mlp(mlp_model):
    return mlp_model.layers

def optimizer_callable(params):
    opt = optim.SGD(params, lr=1e-3)
    return opt

def train_pipe(args, part='parameters'):
    
    deepspeed.runtime.utils.set_random_seed(args.seed)

    #
    # Build the model
    #
    net = MLP(stages=args.pipeline_parallel_size)

    net = PipelineModule(layers=join_layers_mlp(net),
                         loss_fn=torch.nn.CrossEntropyLoss(),
                         num_stages=args.pipeline_parallel_size,
                         partition_method=part,
                         activation_checkpoint_interval=0)

    trainset = cifar_trainset(args.local_rank)

    engine, _, _, _ = deepspeed.initialize(
        args=args,
        model=net,
        optimizer=optimizer_callable,
        model_parameters=[p for p in net.parameters() if p.requires_grad],
        training_data=trainset)
    
    # engine.save_checkpoint(save_dir=args.save_dir, client_state={'checkpoint_step': 0})
    set_seed(123)
    start_time = time.perf_counter()
    for step in range(args.steps):
        loss = engine.train_batch()
    
    engine.save_checkpoint(save_dir=args.save_dir, client_state={'checkpoint_step': step})
    end_time = time.perf_counter()
    execution_time = end_time - start_time
    print(f"Total time for {args.steps} steps (s):", execution_time)

def load_pipe(args, part='parameters'):
    
    deepspeed.runtime.utils.set_random_seed(args.seed)

    #
    # Build the model
    #
    
    net = MLP(stages=args.pipeline_parallel_size)

    net = PipelineModule(layers=join_layers_mlp(net),
                         loss_fn=torch.nn.CrossEntropyLoss(),
                         num_stages=args.pipeline_parallel_size,
                         partition_method=part,
                         activation_checkpoint_interval=0)

    trainset = cifar_trainset(args.local_rank)

    engine, _, _, _ = deepspeed.initialize(
        args=args,
        model=net,
        model_parameters=[p for p in net.parameters() if p.requires_grad],
        training_data=trainset)

    engine.load_checkpoint(args.save_dir, load_module_strict=True)
    print("Model loaded successfully!")

    
if __name__ == '__main__':
    args = get_args()

    deepspeed.init_distributed(dist_backend=args.backend)
    args.local_rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(args.local_rank)


    if args.pipeline_parallel_size == 0:
        train_base(args)
    else:
        train_pipe(args)
