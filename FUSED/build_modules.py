# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import argparse
import datetime
import json



import random
import time
from pathlib import Path
import os, sys
from typing import Optional
from util.get_param_dicts import get_param_dict
from util.logger import setup_logger
import numpy as np
import torch
from torch.utils.data import DataLoader, DistributedSampler
import torch.distributed as dist
import datasets
import util.misc as utils
from datasets import build_dataset, get_coco_api_from_dataset
from engine import evaluate, evaluate_student, evaluate_teacher, train_one_epoch,  test
import models
from util.slconfig import DictAction, SLConfig
from util.utils import ModelEma, BestMetricHolder
from torch.utils.data.sampler import BatchSampler, RandomSampler
from torch.utils.data import DataLoader, DistributedSampler
from datasets.coco_style_dataset import CocoStyleDataset, CocoStyleDatasetTeaching
from datasets.augmentations import weak_aug, strong_aug, base_trans
from torchvision.ops.boxes import nms
from utils.box_utils import box_cxcywh_to_xyxy, generalized_box_iou


def build_model_main(args):
    # we use register to maintain models from catdet6 on.
    from models.registry import MODULE_BUILD_FUNCS
    assert args.modelname in MODULE_BUILD_FUNCS._module_dict
    build_func = MODULE_BUILD_FUNCS.get(args.modelname)
    model, criterion, postprocessors = build_func(args)
    return model, criterion, postprocessors

def build_teacher(args, student_model, device):
    teacher_model, criterion , postprocessors = build_model_main(args)
    state_dict, student_state_dict = teacher_model.state_dict(), student_model.state_dict()
    for key, value in state_dict.items():
        state_dict[key] = student_state_dict[key].clone().detach()
    teacher_model.load_state_dict(state_dict)
    return teacher_model, criterion,  postprocessors


def build_sampler(args, dataset, split):
    if split == 'train':
        if args.distributed:
            sampler = DistributedSampler(dataset, shuffle=True, drop_last=True)
        else:
            sampler = RandomSampler(dataset)
        batch_sampler = BatchSampler(sampler, args.batch_size, drop_last=True)
    else:
        if args.distributed:
            sampler = DistributedSampler(dataset, shuffle=False)
        else:
            sampler = torch.utils.data.SequentialSampler(dataset)
        batch_sampler = BatchSampler(sampler, args.batch_size, drop_last=False)
    return batch_sampler


def build_dataloader(args, dataset_name, domain, split, trans):
    dataset = CocoStyleDataset(root_dir=args.coco_path,
                               dataset_name=dataset_name,
                               domain=domain,
                               split=split,
                               transforms=trans)
    batch_sampler = build_sampler(args, dataset, split)
    data_loader = DataLoader(dataset=dataset,
                             batch_sampler=batch_sampler,
                             collate_fn=CocoStyleDataset.collate_fn,
                             num_workers=args.num_workers)
    return data_loader

def build_dataloader_teaching(args, dataset_name, domain, split):
    dataset = CocoStyleDatasetTeaching(root_dir=args.coco_path,
                                       dataset_name=dataset_name,
                                       domain=domain,
                                       split=split,
                                       weak_aug=weak_aug,
                                       strong_aug=strong_aug,
                                       final_trans=base_trans)
    batch_sampler = build_sampler(args, dataset, split)
    data_loader = DataLoader(dataset=dataset,
                             batch_sampler=batch_sampler,
                             collate_fn=CocoStyleDatasetTeaching.collate_fn_teaching,
                             num_workers=args.num_workers)
    return data_loader

