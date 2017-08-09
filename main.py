from __future__ import print_function

import argparse
import os
import sys

import torch
import torch.optim as optim
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F
from envs import create_atari_env
from model import ActorCritic
from train import train
from test import test
import my_optim

# Based on
# https://github.com/pytorch/examples/tree/master/mnist_hogwild
# Training settings
parser = argparse.ArgumentParser(description='A3C')
parser.add_argument('--lr', type=float, default=0.0001, metavar='LR',
                    help='learning rate (default: 0.0001)')
parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                    help='discount factor for rewards (default: 0.99)')
parser.add_argument('--tau', type=float, default=1.00, metavar='T',
                    help='parameter for GAE (default: 1.00)')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--num-processes', type=int, default=4, metavar='N',
                    help='how many training processes to use (default: 4)')
parser.add_argument('--num-steps', type=int, default=20, metavar='NS',
                    help='number of forward steps in A3C (default: 20)')
parser.add_argument('--max-episode-length', type=int, default=10000, metavar='M',
                    help='maximum length of an episode (default: 10000)')
parser.add_argument('--env-name', default='PongDeterministic-v3', metavar='ENV',
                    help='environment to train on (default: PongDeterministic-v3)')
parser.add_argument('--no-shared', default=False, metavar='O',
                    help='use an optimizer without shared momentum.')

parser.add_argument('--num-skips', type=int, default=3, metavar='SKIP',
                    help='how many frame skip allowed')
parser.add_argument('--model-name', default='def', 
                    help='for saving the model')
parser.add_argument('--load-dir',
                    help='load model from path')
parser.add_argument('--testing', default=False,
                    help='to run model')


if __name__ == '__main__':
    os.environ['OMP_NUM_THREADS'] = '1'  
  
    args = parser.parse_args()
    print(args)

    torch.manual_seed(args.seed)

    env = create_atari_env(args.env_name)
    shared_model = ActorCritic(
        env.observation_space.shape[0], env.action_space, args.num_skips)
    shared_model.share_memory()

    if args.no_shared:
        optimizer = None
    else:
        optimizer = my_optim.SharedAdam(shared_model.parameters(), lr=args.lr)
        optimizer.share_memory()

    if args.load_dir:
        filename = args.load_dir
        print('==> loading checkpoint {}'.format(filename))
        checkpoint = torch.load(filename)
        shared_model.load_state_dict(checkpoint)
        print('==> loaded checkpoint {}'.format(filename))

    processes = []

    p = mp.Process(target=test, args=(args.num_processes, args, shared_model))
    p.start()
    processes.append(p)

    if not args.testing:
        for rank in range(0, args.num_processes):
            p = mp.Process(target=train, args=(rank, args, shared_model, optimizer))
            p.start()
            processes.append(p)
    for p in processes:
        p.join()
