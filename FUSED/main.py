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
from datasets.augmentations import train_trans, val_trans, strong_trans



from util.logger import setup_logger

import numpy as np
import torch
from torch.utils.data import DataLoader, DistributedSampler
import torch.distributed as dist

import datasets
import util.misc as utils
from datasets import build_dataset, get_coco_api_from_dataset
from engine import evaluate, evaluate_student, evaluate_teacher, train_one_epoch, train_one_epoch_teaching_new, test, train_one_epoch_teaching_standard
import models
from util.slconfig import DictAction, SLConfig
from util.utils import ModelEma, BestMetricHolder
from build_modules import *

def get_args_parser():
    parser = argparse.ArgumentParser('Set transformer detector', add_help=False)
    parser.add_argument('--config_file', '-c', type=str, required=True)
    parser.add_argument('--options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file.')

    # dataset parameters
    parser.add_argument('--dataset_file', default='coco_teaching', choices=['coco', 'coco_teaching'])
    parser.add_argument('--coco_path', type=str, default='/comp_robot/cv_public_dataset/COCO2017/')
    parser.add_argument('--coco_panoptic_path', type=str)
    parser.add_argument('--remove_difficult', action='store_true')
    parser.add_argument('--fix_size', action='store_true')


    # training parameters
    parser.add_argument('--output_dir', default='',
                        help='path where to save, empty for no saving')
    parser.add_argument('--note', default='',
                        help='add some notes to the experiment')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--pretrain_model_path', help='load from other checkpoint')
    parser.add_argument('--finetune_ignore', type=str, nargs='+')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true')
    # parser.add_argument('--only_eval', action='store_true')
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--find_unused_params', action='store_true')
    

    parser.add_argument('--save_results', action='store_true')
    parser.add_argument('--save_log', action='store_true')

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--rank', default=0, type=int,
                        help='number of distributed processes')
    parser.add_argument("--local_rank", type=int, help='local rank for DistributedDataParallel')
    parser.add_argument('--amp', action='store_true',
                        help="Train with mixed precision")
    # parser.add_argument('--data_root', default='', type=str)
    parser.add_argument('--target_dataset', default='foggycityscapes', type=str)
    parser.add_argument('--source_dataset', default='cityscapes', type=str)
    parser.add_argument('--expert_embed_dir', default='./expert_embeddings', type=str)
    parser.add_argument('--init_threshold', default=0.4, type=float)
    
    # Expert Arguments
    parser.add_argument('--module_type', default="student_with_no_expert", type=str)  #(student_with_no_expert, expert)
    
    # Hierarchical Domain Adaptation
    parser.add_argument('--hda', default=None, type=int, nargs='+',
                        help="whether to adopt hierarchical domain adaptation (HDA). "
                             "Use layer index 1, 2, 3, 4, 5. The index is shared by both encoder and decoder. "
                             "HDA is turned off when hda is None.")

    # Adapt or source pretrain
    parser.add_argument('--adapt', action='store_true')
    parser.add_argument('--topk_pseudo', type=int, default=10)
    
    # Contrastive Knowledge Distillation Hyperparameters (memory_size, temp)
    parser.add_argument('--temp', default=1.0, type=float)
    parser.add_argument('--memory_bank_size', default=1, type=int)
    parser.add_argument('--lambda_eckd', default=1.0, type=float)
    parser.add_argument('--conf_update_algo', default="raise_abruptly", 
                        choices=['raise_slowly', 'raise_abruptly', "const_thresh"], type=str)
    
    return parser


def build_model_main(args):
    # we use register to maintain models from catdet6 on.
    from models.registry import MODULE_BUILD_FUNCS
    assert args.modelname in MODULE_BUILD_FUNCS._module_dict
    build_func = MODULE_BUILD_FUNCS.get(args.modelname)
    model, criterion, postprocessors = build_func(args)
    return model, criterion, postprocessors

def calculate_confidence(confidence_list):
    # Flatten the nested list into a 1D list
    flat_confidence_list = [item for sublist in confidence_list for item in sublist]

    # Calculate the average confidence
    average_confidence = torch.mean(torch.tensor(flat_confidence_list))
    
    # Find the highest confidence
    highest_confidence = torch.max(torch.tensor(flat_confidence_list))

    # Calculate the average of both average and highest confidence
    combined_average = (average_confidence + highest_confidence) / 2

    return combined_average.item()

def fancy_calculate_confidence(confidence_list, epoch_num, previous_threshold):
    # Flatten the nested list into a 1D list
    flat_confidence_list = [item for sublist in confidence_list for item in sublist]

    # Calculate the average confidence
    average_confidence = torch.mean(torch.tensor(flat_confidence_list))
    
    # Find the highest confidence
    highest_confidence = torch.max(torch.tensor(flat_confidence_list))

    # Calculate the average of both average and highest confidence
    combined_average = (average_confidence + highest_confidence) / 2

    if epoch_num == 0:
        return combined_average.item() / 6
    else:
        return previous_threshold




def main(args):
    utils.init_distributed_mode(args)
    # load cfg file and update the args
    print("Loading config file from {}".format(args.config_file))
    time.sleep(args.rank * 0.02)
    cfg = SLConfig.fromfile(args.config_file)
    if args.options is not None:
        cfg.merge_from_dict(args.options)
    if args.rank == 0:
        save_cfg_path = os.path.join(args.output_dir, "config_cfg.py")
        cfg.dump(save_cfg_path)
        save_json_path = os.path.join(args.output_dir, "config_args_raw.json")
        with open(save_json_path, 'w') as f:
            json.dump(vars(args), f, indent=2)
    cfg_dict = cfg._cfg_dict.to_dict()
    args_vars = vars(args)
    for k,v in cfg_dict.items():
        if k not in args_vars:
            setattr(args, k, v)
        else:
            raise ValueError("Key {} can used by args only".format(k))

    # update some new args temporally
    if not getattr(args, 'use_ema', None):
        args.use_ema = False
    if not getattr(args, 'debug', None):
        args.debug = False

    # setup logger
    os.makedirs(args.output_dir, exist_ok=True)
    logger = setup_logger(output=os.path.join(args.output_dir, 'info.txt'), distributed_rank=args.rank, color=False, name="detr")
    logger.info("git:\n  {}\n".format(utils.get_sha()))
    logger.info("Command: "+' '.join(sys.argv))
    if args.rank == 0:
        save_json_path = os.path.join(args.output_dir, "config_args_all.json")
        with open(save_json_path, 'w') as f:
            json.dump(vars(args), f, indent=2)
        logger.info("Full config saved to {}".format(save_json_path))
    logger.info('world size: {}'.format(args.world_size))
    logger.info('rank: {}'.format(args.rank))
    logger.info('local_rank: {}'.format(args.local_rank))
    logger.info("args: " + str(args) + '\n')

    if args.frozen_weights is not None:
        assert args.masks, "Frozen training is meant for segmentation only"
    print(args)

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed) 
    torch.cuda.manual_seed(seed)  # If using CUDA

    # build model
    model, criterion, postprocessors = build_model_main(args)
    wo_class_error = False
    model.to(device)

    # ema
    if args.use_ema:
        ema_m = ModelEma(model, args.ema_decay)
    else:
        ema_m = None

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=args.find_unused_params)
        model_without_ddp = model.module
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info('number of params:'+str(n_parameters))
    logger.info("params:\n"+json.dumps({n: p.numel() for n, p in model.named_parameters() if p.requires_grad}, indent=2))

    param_dicts = get_param_dict(args, model_without_ddp)

    optimizer = torch.optim.AdamW(param_dicts, lr=args.lr,
                                  weight_decay=args.weight_decay)
    
    dataset_train = build_dataset(image_set='train', args=args)
    dataset_val = build_dataset(image_set='val', args=args)

    if args.distributed:
        sampler_train = DistributedSampler(dataset_train)
        sampler_val = DistributedSampler(dataset_val, shuffle=False)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    batch_sampler_train = torch.utils.data.BatchSampler(
        sampler_train, args.batch_size, drop_last=True)

    data_loader_train = DataLoader(dataset_train, batch_sampler=batch_sampler_train,
                                   collate_fn=utils.collate_fn, num_workers=args.num_workers)
    data_loader_val = DataLoader(dataset_val, 1, sampler=sampler_val,
                                 drop_last=False, collate_fn=utils.collate_fn, num_workers=args.num_workers)

    if args.onecyclelr:
        lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.lr, steps_per_epoch=len(data_loader_train), epochs=args.epochs, pct_start=0.2)
    elif args.multi_step_lr:
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.lr_drop_list)
    else:
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop)


    if args.dataset_file == "coco_panoptic":
        # We also evaluate AP during panoptic training, on original coco DS
        coco_val = datasets.coco.build("val", args)
        base_ds = get_coco_api_from_dataset(coco_val)
    else:
        base_ds = get_coco_api_from_dataset(dataset_val)

    if args.frozen_weights is not None:
        checkpoint = torch.load(args.frozen_weights, map_location='cpu')
        model_without_ddp.detr.load_state_dict(checkpoint['model'])

    output_dir = Path(args.output_dir)
    if os.path.exists(os.path.join(args.output_dir, 'checkpoint.pth')):
        args.resume = os.path.join(args.output_dir, 'checkpoint.pth')
    if args.resume:
        if args.resume.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.resume, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.resume, map_location='cpu')
        model_without_ddp.load_state_dict(checkpoint['model'])
        if args.use_ema:
            if 'ema_model' in checkpoint:
                ema_m.module.load_state_dict(utils.clean_state_dict(checkpoint['ema_model']))
            else:
                del ema_m
                ema_m = ModelEma(model, args.ema_decay)                

        if not args.eval and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            args.start_epoch = checkpoint['epoch'] + 1

    if (not args.resume) and args.pretrain_model_path:
        checkpoint = torch.load(args.pretrain_model_path, map_location='cpu')['model']
        from collections import OrderedDict
        _ignorekeywordlist = args.finetune_ignore if args.finetune_ignore else []
        ignorelist = []

        def check_keep(keyname, ignorekeywordlist):
            for keyword in ignorekeywordlist:
                if keyword in keyname:
                    ignorelist.append(keyname)
                    return False
            return True

        logger.info("Ignore keys: {}".format(json.dumps(ignorelist, indent=2)))
        _tmp_st = OrderedDict({k:v for k, v in utils.clean_state_dict(checkpoint).items() if check_keep(k, _ignorekeywordlist)})

        _load_output = model_without_ddp.load_state_dict(_tmp_st, strict=False)
        logger.info(str(_load_output))

        if args.use_ema:
            if 'ema_model' in checkpoint:
                ema_m.module.load_state_dict(utils.clean_state_dict(checkpoint['ema_model']))
            else:
                del ema_m
                ema_m = ModelEma(model, args.ema_decay)        

    if args.eval:
        os.environ['EVAL_FLAG'] = 'TRUE'
        test_stats, coco_evaluator = evaluate(model, criterion, postprocessors,
                                              data_loader_val, base_ds, device, args.output_dir, wo_class_error=wo_class_error, args=args)
        if args.output_dir:
            utils.save_on_master(coco_evaluator.coco_eval["bbox"].eval, output_dir / "eval.pth")

        log_stats = {**{f'test_{k}': v for k, v in test_stats.items()} }
        if args.output_dir and utils.is_main_process():
            with (output_dir / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")

        return

    print("Start training")
    start_time = time.time()
    best_map_holder = BestMetricHolder(use_ema=args.use_ema)
    for epoch in range(args.start_epoch, args.epochs):
        epoch_start_time = time.time()
        if args.distributed:
            sampler_train.set_epoch(epoch)
        train_stats = train_one_epoch(
            model, criterion, data_loader_train, optimizer, device, epoch,
            args.clip_max_norm, wo_class_error=wo_class_error, lr_scheduler=lr_scheduler, args=args, logger=(logger if args.save_log else None), ema_m=ema_m)
        if args.output_dir:
            checkpoint_paths = [output_dir / 'checkpoint.pth']

        if not args.onecyclelr:
            lr_scheduler.step()
        if args.output_dir:
            checkpoint_paths = [output_dir / 'checkpoint.pth']
            # extra checkpoint before LR drop and every 100 epochs
            if (epoch + 1) % args.lr_drop == 0 or (epoch + 1) % args.save_checkpoint_interval == 0:
                checkpoint_paths.append(output_dir / f'checkpoint{epoch:04}.pth')
            for checkpoint_path in checkpoint_paths:
                weights = {
                    'model': model_without_ddp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'epoch': epoch,
                    'args': args,
                }
                if args.use_ema:
                    weights.update({
                        'ema_model': ema_m.module.state_dict(),
                    })
                utils.save_on_master(weights, checkpoint_path)
                
        # eval
        test_stats, coco_evaluator = evaluate(
            model, criterion, postprocessors, data_loader_val, base_ds, device, args.output_dir,
            wo_class_error=wo_class_error, args=args, logger=(logger if args.save_log else None)
        )
        map_regular = test_stats['coco_eval_bbox'][0]
        _isbest = best_map_holder.update(map_regular, epoch, is_ema=False)
        if _isbest:
            checkpoint_path = output_dir / 'checkpoint_best_regular.pth'
            utils.save_on_master({
                'model': model_without_ddp.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                'epoch': epoch,
                'args': args,
            }, checkpoint_path)
        log_stats = {
            **{f'train_{k}': v for k, v in train_stats.items()},
            **{f'test_{k}': v for k, v in test_stats.items()},
        }

        # eval ema
        if args.use_ema:
            ema_test_stats, ema_coco_evaluator = evaluate(
                ema_m.module, criterion, postprocessors, data_loader_val, base_ds, device, args.output_dir,
                wo_class_error=wo_class_error, args=args, logger=(logger if args.save_log else None)
            )
            log_stats.update({f'ema_test_{k}': v for k,v in ema_test_stats.items()})
            map_ema = ema_test_stats['coco_eval_bbox'][0]
            _isbest = best_map_holder.update(map_ema, epoch, is_ema=True)
            if _isbest:
                checkpoint_path = output_dir / 'checkpoint_best_ema.pth'
                utils.save_on_master({
                    'model': ema_m.module.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'epoch': epoch,
                    'args': args,
                }, checkpoint_path)
        log_stats.update(best_map_holder.summary())

        ep_paras = {
                'epoch': epoch,
                'n_parameters': n_parameters
            }
        log_stats.update(ep_paras)
        try:
            log_stats.update({'now_time': str(datetime.datetime.now())})
        except:
            pass
        
        epoch_time = time.time() - epoch_start_time
        epoch_time_str = str(datetime.timedelta(seconds=int(epoch_time)))
        log_stats['epoch_time'] = epoch_time_str

        if args.output_dir and utils.is_main_process():
            with (output_dir / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")

            # for evaluation logs
            if coco_evaluator is not None:
                (output_dir / 'eval').mkdir(exist_ok=True)
                if "bbox" in coco_evaluator.coco_eval:
                    filenames = ['latest.pth']
                    if epoch % 50 == 0:
                        filenames.append(f'{epoch:03}.pth')
                    for name in filenames:
                        torch.save(coco_evaluator.coco_eval["bbox"].eval,
                                   output_dir / "eval" / name)
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))

    # remove the copied files.
    copyfilelist = vars(args).get('copyfilelist')
    if copyfilelist and args.local_rank == 0:
        from datasets.data_util import remove
        for filename in copyfilelist:
            print("Removing: {}".format(filename))
            remove(filename)


def teaching_new(args):
    utils.init_distributed_mode(args)
    # load cfg file and update the args
    print("Loading config file from {}".format(args.config_file))
    time.sleep(args.rank * 0.02)
    cfg = SLConfig.fromfile(args.config_file)
    if args.options is not None:
        cfg.merge_from_dict(args.options)
    if args.rank == 0:
        save_cfg_path = os.path.join(args.output_dir, "config_cfg.py")
        cfg.dump(save_cfg_path)
        save_json_path = os.path.join(args.output_dir, "config_args_raw.json")
        with open(save_json_path, 'w') as f:
            json.dump(vars(args), f, indent=2)
    cfg_dict = cfg._cfg_dict.to_dict()
    args_vars = vars(args)
    for k,v in cfg_dict.items():
        if k not in args_vars:
            setattr(args, k, v)
        else:
            raise ValueError("Key {} can used by args only".format(k))

    # update some new args temporally
    if not getattr(args, 'use_ema', None):
        args.use_ema = False
    if not getattr(args, 'debug', None):
        args.debug = False

    # setup logger
    os.makedirs(args.output_dir, exist_ok=True)
    logger = setup_logger(output=os.path.join(args.output_dir, 'info.txt'), distributed_rank=args.rank, color=False, name="detr")
    logger.info("git:\n  {}\n".format(utils.get_sha()))
    logger.info("Command: "+' '.join(sys.argv))
    if args.rank == 0:
        save_json_path = os.path.join(args.output_dir, "config_args_all.json")
        with open(save_json_path, 'w') as f:
            json.dump(vars(args), f, indent=2)
        logger.info("Full config saved to {}".format(save_json_path))
    logger.info('world size: {}'.format(args.world_size))
    logger.info('rank: {}'.format(args.rank))
    logger.info('local_rank: {}'.format(args.local_rank))
    logger.info("args: " + str(args) + '\n')


    if args.frozen_weights is not None:
        assert args.masks, "Frozen training is meant for segmentation only"
    print(args)

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # build student and teacher model
    student_model, student_criterion, student_postprocessors = build_model_main(args)
    teacher_model, teacher_criterion, teacher_postprocessors = build_teacher(args, student_model, device)
    wo_class_error = False

    # BUILD THESE EMBEDDERS
    if args.module_type != "student_with_no_expert":
        student_model.build_expert_embedder(device)
        # For student embedder, change the in_channels_list according to the architecture
        # student_model.build_backbone_embedder(in_channels_list=[384, 768, 1536], out_dim=768, device=device)
        student_model.build_backbone_embedder(in_channels_list=[256, 256, 256], out_dim=768, device=device)
        # student_model.build_expert_assistant(device)
    
    student_model.to(device)
    teacher_model.to(device)

    # ema
    if args.use_ema:
        ema_m = ModelEma(teacher_model, args.ema_decay)
    else:
        ema_m = None

    student_model_without_ddp = student_model
    teacher_model_without_ddp = teacher_model
    if args.distributed:
        student_model = torch.nn.parallel.DistributedDataParallel(student_model, device_ids=[args.gpu], find_unused_parameters=args.find_unused_params)
        teacher_model = torch.nn.parallel.DistributedDataParallel(teacher_model, device_ids=[args.gpu])
        student_model._set_static_graph()
        teacher_model._set_static_graph()
        student_model_without_ddp = student_model.module
        teacher_model_without_ddp = teacher_model.module
        
    n_parameters_student = sum(p.numel() for p in student_model.parameters() if p.requires_grad)
    n_parameters_teacher = sum(p.numel() for p in teacher_model.parameters() if p.requires_grad)
    logger.info('number of student params:'+str(n_parameters_student))
    logger.info('number of teacher params:'+str(n_parameters_teacher))
    logger.info("params student:\n"+json.dumps({n: p.numel() for n, p in student_model.named_parameters() if p.requires_grad}, indent=2))
    logger.info("params teacher:\n"+json.dumps({n: p.numel() for n, p in teacher_model.named_parameters() if p.requires_grad}, indent=2))

    param_dicts_student = get_param_dict(args, student_model_without_ddp)
   

    optimizer_student = torch.optim.AdamW(param_dicts_student, lr=args.lr,
                                  weight_decay=args.weight_decay)
    
    
    
    target_loader = build_dataloader_teaching(args, args.target_dataset, 'target', 'test')

    



    if args.onecyclelr:
        lr_scheduler_student = torch.optim.lr_scheduler.OneCycleLR(optimizer_student, max_lr=args.lr, steps_per_epoch=len(target_loader), epochs=args.epochs, pct_start=0.2)
   
    elif args.multi_step_lr:
        lr_scheduler_student = torch.optim.lr_scheduler.MultiStepLR(optimizer_student, milestones=args.lr_drop_list)
        
    else:
        lr_scheduler_student = torch.optim.lr_scheduler.StepLR(optimizer_student, args.lr_drop)
        



    if args.dataset_file == "coco_panoptic":
        # We also evaluate AP during panoptic training, on original coco DS
        coco_val = datasets.coco.build("val", args)
        base_ds = get_coco_api_from_dataset(coco_val)
    else:
        base_ds_student = get_coco_api_from_dataset(target_loader)
        base_ds_teacher = get_coco_api_from_dataset(target_loader)
        

    if args.frozen_weights is not None:
        checkpoint_student = torch.load(args.frozen_weights, map_location='cpu')
        student_model_without_ddp.detr.load_state_dict(checkpoint_student['model'])
        checkpoint_teacher = torch.load(args.frozen_weights, map_location='cpu')
        teacher_model_without_ddp.detr.load_state_dict(checkpoint_teacher['model'])

    output_dir = Path(args.output_dir)
    if os.path.exists(os.path.join(args.output_dir, 'checkpoint.pth')):
        args.resume = os.path.join(args.output_dir, 'checkpoint.pth')
    if args.resume:
        if args.resume.startswith('https'):
            checkpoint_student = torch.hub.load_state_dict_from_url(
                args.resume, map_location='cpu', check_hash=True)
            checkpoint_teacher = torch.hub.load_state_dict_from_url(
                args.resume, map_location='cpu', check_hash=True)
        else:
            checkpoint_student = torch.load(args.resume, map_location='cpu')
            checkpoint_teacher = torch.load(args.resume, map_location='cpu')
        student_model_without_ddp.load_state_dict(checkpoint_student['model'])
        teacher_model_without_ddp.load_state_dict(checkpoint_teacher['model'])
        if args.use_ema:
            if 'ema_model' in checkpoint_teacher:
                ema_m.module.load_state_dict(utils.clean_state_dict(checkpoint_teacher['ema_model']))
            else:
                del ema_m
                ema_m = ModelEma(student_model, args.ema_decay)                

        if not args.eval and 'optimizer' in checkpoint_student and 'lr_scheduler' in checkpoint_student and 'epoch' in checkpoint_student:
            optimizer_student.load_state_dict(checkpoint_student['optimizer'])
            lr_scheduler_student.load_state_dict(checkpoint_student['lr_scheduler'])
            args.start_epoch = checkpoint_student['epoch'] + 1
  
            

    if (not args.resume) and args.pretrain_model_path:
        checkpoint_student = torch.load(args.pretrain_model_path, map_location='cpu')['model']
        checkpoint_teacher = torch.load(args.pretrain_model_path, map_location='cpu')['model']
        from collections import OrderedDict
        _ignorekeywordlist = args.finetune_ignore if args.finetune_ignore else []
        ignorelist = []

        def check_keep(keyname, ignorekeywordlist):
            for keyword in ignorekeywordlist:
                if keyword in keyname:
                    ignorelist.append(keyname)
                    return False
            return True

        logger.info("Ignore keys: {}".format(json.dumps(ignorelist, indent=2)))
        _tmp_st_student = OrderedDict({k:v for k, v in utils.clean_state_dict(checkpoint_student).items() if check_keep(k, _ignorekeywordlist)})
        _tmp_st_teacher = OrderedDict({k:v for k, v in utils.clean_state_dict(checkpoint_teacher).items() if check_keep(k, _ignorekeywordlist)})

        _load_output_student = student_model_without_ddp.load_state_dict(_tmp_st_student, strict=False)
        _load_output_teacher = teacher_model_without_ddp.load_state_dict(_tmp_st_teacher, strict=False)
        
        logger.info(str(_load_output_student))
        logger.info(str(_load_output_teacher))
        
        

              


    threshold = args.init_threshold
    if not args.eval:
        print("Started teacher training")
        start_time = time.time()
        best_map_holder = BestMetricHolder(use_ema=args.use_ema)
        keep_th_const = False
        for epoch in range(args.start_epoch, args.epochs):
            epoch_start_time = time.time()
            if args.distributed and hasattr(target_loader.sampler, 'set_epoch'):
                target_loader.set_epoch(epoch)
                
            if args.module_type == "expert":
                target_stats, confidences = train_one_epoch_teaching_new(
                    student_model, teacher_model, student_criterion, teacher_criterion,  target_loader, optimizer_student, device, epoch, threshold,
                    args.clip_max_norm, wo_class_error=wo_class_error, lr_scheduler_student=lr_scheduler_student, args=args, logger=(logger if args.save_log else None), ema_m=ema_m)
                threshold = calculate_confidence(confidences)
                
                    
            else:
                target_stats, confidences = train_one_epoch_teaching_standard(
                    student_model, teacher_model, student_criterion, teacher_criterion,  target_loader, optimizer_student, device, epoch, threshold,
                    args.clip_max_norm, wo_class_error=wo_class_error, lr_scheduler_student=lr_scheduler_student, args=args, logger=(logger if args.save_log else None), ema_m=ema_m)
                
                threshold = threshold
                
                

            if args.output_dir:
                checkpoint_paths_student = [output_dir / 'checkpoint_student.pth']
                checkpoint_paths_teacher = [output_dir / 'checkpoint_teacher.pth']

            if not args.onecyclelr:
                lr_scheduler_student.step()

            if args.output_dir:
                checkpoint_paths_student = [output_dir / 'checkpoint_student.pth']
                checkpoint_paths_teacher = [output_dir / 'checkpoint_teacher.pth']
                # extra checkpoint before LR drop and every 100 epochs
                if (epoch + 1) % args.lr_drop == 0 or (epoch + 1) % args.save_checkpoint_interval == 0:
                    checkpoint_paths_student.append(output_dir / f'student_checkpoint{epoch:04}.pth')
                    checkpoint_paths_teacher.append(output_dir / f'teacher_checkpoint{epoch:04}.pth')
                for checkpoint_path in checkpoint_paths_student:
                    weights = {
                        'model': student_model_without_ddp.state_dict(),
                        'optimizer':optimizer_student.state_dict(),
                        'lr_scheduler': lr_scheduler_student.state_dict(),
                        'epoch': epoch,
                        'args': args,
                    }
                    if args.use_ema:
                        weights.update({
                            'ema_model': ema_m.module.state_dict(),
                        })
                    utils.save_on_master(weights, checkpoint_path)
                for checkpoint_path in checkpoint_paths_teacher:
                    weights = {
                        'model': teacher_model_without_ddp.state_dict(),
                        'epoch': epoch,
                        'args': args,
                    }
                    if args.use_ema:
                        weights.update({
                            'ema_model': ema_m.module.state_dict(),
                        })
                    utils.save_on_master(weights, checkpoint_path)
                    
            # eval
            # test_stats, coco_evaluator = evaluate_student(
            #     student_model, student_criterion, student_postprocessors, target_loader, base_ds, device, args.output_dir,
            #     wo_class_error=wo_class_error, args=args, logger=(logger if args.save_log else None)
            # )
            # map_regular = test_stats['coco_eval_bbox'][0]
            # _isbest = best_map_holder.update(map_regular, epoch, is_ema=False)
            # if _isbest:
            #     checkpoint_path = output_dir / 'checkpoint_best_regular.pth'
            #     utils.save_on_master({
            #         'model': student_model_without_ddp.state_dict(),
            #         'optimizer': optimizer_student.state_dict(),
            #         'lr_scheduler': lr_scheduler_student.state_dict(),
            #         'epoch': epoch,
            #         'args': args,
            #     }, checkpoint_path)
            # log_stats = {
            #     **{f'train_{k}': v for k, v in train_stats.items()},
            #     **{f'test_{k}': v for k, v in test_stats.items()},
            # }
            
            
            # eval ema
            
            

            ep_paras_student = {
                    'epoch': epoch,
                    'n_parameters': n_parameters_student
                }
    else:       
            
        test_stats, coco_evaluator = evaluate_student(
            student_model, student_criterion, student_postprocessors, target_loader, base_ds_student, device, args.output_dir,
            wo_class_error=wo_class_error, args=args, logger=(logger if args.save_log else None)
        )
        map_regular = test_stats['coco_eval_bbox'][0]
        _isbest = best_map_holder.update(map_regular, epoch, is_ema=False)
        if _isbest:
            checkpoint_path = output_dir / 'checkpoint_best_regular.pth'
            utils.save_on_master({
                'model': student_model_without_ddp.state_dict(),
                'optimizer': optimizer_student.state_dict(),
                'lr_scheduler': lr_scheduler_student.state_dict(),
                'epoch': epoch,
                'args': args,
            }, checkpoint_path)
        log_stats = {
            # **{f'train_{k}': v for k, v in train_stats.items()},
            **{f'test_{k}': v for k, v in test_stats.items()},
        }

           
                        
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time for student {}'.format(total_time_str))
    print('Training time for teacher {}'.format(total_time_str))

    # remove the copied files.
    copyfilelist = vars(args).get('copyfilelist')
    if copyfilelist and args.local_rank == 0:
        from datasets.data_util import remove
        for filename in copyfilelist:
            print("Removing: {}".format(filename))
            remove(filename)
        
  
            
if __name__ == '__main__':
    parser = argparse.ArgumentParser('DETR training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    if args.adapt:
        teaching_new(args)
    else:
        main(args)