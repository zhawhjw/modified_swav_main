import argparse
import math
import os
import shutil
import time
from logging import getLogger

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import apex
from apex.parallel.LARC import LARC

from src.utils import (
    bool_flag,
    initialize_exp,
    restart_from_checkpoint,
    fix_random_seeds,
    AverageMeter,
    init_distributed_mode,
)
from src.multicropdataset import MultiCropDataset
import src.resnet50 as resnet_models

from modified_main_swav import main

prototypes = [500, 1000, 2000, 3000]
batch_size = 32
epochs = 1
qstart = 90
lr = 0.7

directory = ['checkpoints']
file = ["checkpoint.pth.tar", "params.pkl", "stats0.pkl", "train.log", "queue0.pth"]

saved_path = "saved"
if not os.path.exists(saved_path):
    os.mkdir(saved_path)


def wrap():
    logger = getLogger()

    parser = argparse.ArgumentParser(description="Implementation of SwAV")

    #########################
    #### data parameters ####
    #########################
    parser.add_argument("--data_path", type=str, default="/path/to/imagenet",
                        help="path to dataset repository")
    parser.add_argument("--nmb_crops", type=int, default=[2], nargs="+",
                        help="list of number of crops (example: [2, 6])")
    parser.add_argument("--size_crops", type=int, default=[224], nargs="+",
                        help="crops resolutions (example: [224, 96])")
    parser.add_argument("--min_scale_crops", type=float, default=[0.14], nargs="+",
                        help="argument in RandomResizedCrop (example: [0.14, 0.05])")
    parser.add_argument("--max_scale_crops", type=float, default=[1], nargs="+",
                        help="argument in RandomResizedCrop (example: [1., 0.14])")

    #########################
    ## swav specific params #
    #########################
    parser.add_argument("--crops_for_assign", type=int, nargs="+", default=[0, 1],
                        help="list of crops id used for computing assignments")
    parser.add_argument("--temperature", default=0.1, type=float,
                        help="temperature parameter in training loss")
    parser.add_argument("--epsilon", default=0.05, type=float,
                        help="regularization parameter for Sinkhorn-Knopp algorithm")
    parser.add_argument("--sinkhorn_iterations", default=3, type=int,
                        help="number of iterations in Sinkhorn-Knopp algorithm")
    parser.add_argument("--feat_dim", default=128, type=int,
                        help="feature dimension")
    parser.add_argument("--nmb_prototypes", default=3000, type=int,
                        help="number of prototypes")
    parser.add_argument("--queue_length", type=int, default=0,
                        help="length of the queue (0 for no queue)")
    parser.add_argument("--epoch_queue_starts", type=int, default=15,
                        help="from this epoch, we start using a queue")

    #########################
    #### optim parameters ###
    #########################
    parser.add_argument("--epochs", default=800, type=int,
                        help="number of total epochs to run")
    parser.add_argument("--batch_size", default=64, type=int,
                        help="batch size per gpu, i.0.6_lr_500_queue_90_qstart_500_prototype. how many unique instances per gpu")
    parser.add_argument("--base_lr", default=4.8, type=float, help="base learning rate")
    parser.add_argument("--final_lr", type=float, default=0, help="final learning rate")
    parser.add_argument("--freeze_prototypes_niters", default=313, type=int,
                        help="freeze the prototypes during this many iterations from the start")
    parser.add_argument("--wd", default=1e-6, type=float, help="weight decay")
    parser.add_argument("--warmup_epochs", default=10, type=int, help="number of warmup epochs")
    parser.add_argument("--start_warmup", default=0, type=float,
                        help="initial warmup learning rate")

    #########################
    #### dist parameters ###
    #########################
    parser.add_argument("--dist_url", default="env://", type=str, help="""url used to set up distributed
                        training; see https://pytorch.org/docs/stable/distributed.html""")
    parser.add_argument("--world_size", default=-1, type=int, help="""
                        number of processes: it is set automatically and
                        should not be passed as argument""")
    parser.add_argument("--rank", default=0, type=int, help="""rank of this process:
                        it is set automatically and should not be passed as argument""")
    parser.add_argument("--local_rank", default=0, type=int,
                        help="this argument is not used and should be ignored")

    #########################
    #### other parameters ###
    #########################
    parser.add_argument("--arch", default="resnet50", type=str, help="convnet architecture")
    parser.add_argument("--hidden_mlp", default=2048, type=int,
                        help="hidden layer dimension in projection head")
    parser.add_argument("--workers", default=10, type=int,
                        help="number of data loading workers")
    parser.add_argument("--checkpoint_freq", type=int, default=25,
                        help="Save the model periodically")
    parser.add_argument("--use_fp16", type=bool_flag, default=True,
                        help="whether to train with mixed precision or not")
    parser.add_argument("--sync_bn", type=str, default="pytorch", help="synchronize bn")
    parser.add_argument("--syncbn_process_group_size", type=int, default=8, help=""" see
                        https://github.com/NVIDIA/apex/blob/master/apex/parallel/__init__.py#L58-L67""")
    parser.add_argument("--dump_path", type=str, default=".",
                        help="experiment dump path for checkpoints and log")
    parser.add_argument("--seed", type=int, default=31, help="seed")

    args = parser.parse_args()
    args.dist_url = "file:///D:/Github/swav/somefile.txt"

    init_distributed_mode(args)
    fix_random_seeds(args.seed)

    for pro in prototypes:



        args.data_path = "D:\\Data\\3D-FUTURE-model\\train"
        args.epochs = epochs
        args.base_lr = lr
        args.queue_length = pro
        args.nmb_prototypes = pro
        args.epoch_queue_starts = qstart


        target_path = str(args.base_lr) + "_lr_" + \
                      str(args.queue_length) + "_queue_" + \
                      str(args.epoch_queue_starts) + "_qstart_" + \
                      str(args.nmb_prototypes) + "_prototype"
        if not os.path.exists(saved_path + "/" + target_path):
            os.mkdir(saved_path + "/" + target_path)

        args.dump_path = saved_path + "/" + target_path

        args.final_lr = 0.0006
        args.warmup_epochs = 0
        args.batch_size = batch_size
        args.size_crops = [224, 96]
        args.nmb_crops = [2, 6]
        args.min_scale_crops = [0.14, 0.05]
        args.max_scale_crops = [1., 0.14]
        args.use_fp16 = True
        args.freeze_prototypes_niters = 5005

        # args.dist_url = "file:///D:/Github/swav/somefile.txt"

        logger, training_stats = initialize_exp(args, "epoch", "loss")

        main(args, logger, training_stats)
        # "0.6_lr_500_queue_90_qstart_500_prototype"






if __name__ == "__main__":
    extend_dict = {'MASTER_ADDR': '127.0.0.1', 'MASTER_PORT': '29500', 'WORLD_SIZE': '1', 'RANK': '0',
                   'LOCAL_RANK': '0'}
    os.environ.update(extend_dict)
    wrap()
