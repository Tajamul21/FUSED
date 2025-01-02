# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Train and eval functions used in main.py
"""

import math
import os
import sys
from typing import Iterable
import csv
from util.utils import slprint, to_device

import torch
import numpy as np

import util.misc as utils
# from SFDA.Proposed.FocalNet_DINO.datasets.coco_eval_without import CocoEvaluator
from datasets.coco_eval import CocoEvaluator
from datasets.panoptic_eval import PanopticEvaluator
from datasets.coco_style_dataset import DataPreFetcher
from collections import defaultdict
from torchvision.ops.boxes import nms
from utils.box_utils import box_cxcywh_to_xyxy, generalized_box_iou
from typing import List
from torchvision.ops import nms
from torchvision.utils import save_image



def fusion_function(pseudo_labels, matched_predictions):
    filtered_labels = []
    image_ids_with_prediction_1 = [mp['image_id'] for mp in matched_predictions if mp['prediction'] == 1]

    for label in pseudo_labels:
        if label and 'image_id' in label and label['image_id'] in image_ids_with_prediction_1:
            filtered_labels.append(label)

    print(f'Number of filtered pseudo labels: {len(filtered_labels)}')
    return filtered_labels




def match_predictions(csv_file, annotations):
    predictions = []

    with open(csv_file, 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            predictions.append({'image_id': int(row['image_id']), 'prediction': int(row['cancer_present'])})

    matched_predictions = []

    for prediction in predictions:
        image_id = prediction['image_id']
        for annotation in annotations:
            if annotation['image_id'].item() == image_id:
                matched_predictions.append({'image_id': image_id, 'prediction': prediction['prediction']})
                break  # Stop searching once a match is found
            
    return matched_predictions



def create_pseudo_labels(outputs_teacher, threshold, iou_threshold=-1,top_k=100):
    pseudo_labels = []
    batch_counts = {}
    highest_confidences = []

    for batch_idx in range(len(outputs_teacher['pred_logits'])):
        pred_logits = outputs_teacher['pred_logits'][batch_idx]
        pred_boxes = outputs_teacher['pred_boxes'][batch_idx]

        # Get the top-k predictions based on the confidence score
        top_k_indices = pred_logits[:, 0].argsort(descending=True)[:top_k]
        top_k_logits = pred_logits[top_k_indices]
        top_k_boxes = pred_boxes[top_k_indices]

        # Get the highest confidence score
        highest_confidence = top_k_logits[0, 0].sigmoid().item() if len(top_k_logits) > 0 else None
        highest_confidences.append(highest_confidence)
        print(f'Highest Confidence in Batch Index {batch_idx}: {highest_confidence}')

        # Filter based on the threshold
        mask = top_k_logits[:, 0].sigmoid() > threshold  # Assuming the class index for cancer is 0
        filtered_logits = top_k_logits[mask]
        filtered_boxes = top_k_boxes[mask]


        batch_pseudo_labels = {
            'boxes': torch.empty(0, 4, device='cuda'),  # Empty tensor for boxes with CUDA device
            'labels': torch.empty(0, dtype=torch.int64, device='cuda'),  # Empty tensor for labels with CUDA device
            'area': torch.empty(0, device='cuda'),  # Empty tensor for area with CUDA device
            'iscrowd': torch.empty(0, dtype=torch.int64, device='cuda'),  # Empty tensor for iscrowd with CUDA device
            'orig_size': torch.empty(0, dtype=torch.int64, device='cuda'),  # Tensor for orig_size with CUDA device
            'size': torch.empty(0, dtype=torch.int64, device='cuda')  # Tensor for size with CUDA device
        }

        if len(filtered_boxes) > 0:
            # Perform Non-Maximum Suppression (NMS)
            keep_indices = nms(filtered_boxes, filtered_logits[:, 0], iou_threshold)

         

            # Update batch_pseudo_labels based on the filtered and NMS-processed results
            batch_pseudo_labels['boxes'] = filtered_boxes[keep_indices].to('cuda')
            batch_pseudo_labels['labels'] = torch.zeros(len(keep_indices), dtype=torch.int64, device='cuda')  # Assign label 0 with CUDA device

        # Include additional information from outputs_teacher in batch_pseudo_labels
        for key in ['image_id', 'area', 'iscrowd', 'orig_size', 'size']:
            if key in outputs_teacher:
                batch_pseudo_labels[key] = outputs_teacher[key][batch_idx].unsqueeze(0).to('cuda')

        pseudo_labels.append(batch_pseudo_labels)
        # Count the number of pseudo labels for each batch index
        batch_counts[batch_idx] = len(batch_pseudo_labels['boxes'])

    # Print the number of pseudo labels for each batch index
    for batch_idx, count in batch_counts.items():
        print(f'Batch Index {batch_idx}: {count} pseudo labels')

    return pseudo_labels, highest_confidences



def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, max_norm: float = 0, 
                    wo_class_error=False, lr_scheduler=None, args=None, logger=None, ema_m=None):
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)

    try:
        need_tgt_for_training = args.use_dn
    except:
        need_tgt_for_training = False

    model.train()
    criterion.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    if not wo_class_error:
        metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    _cnt = 0
    for samples, targets in metric_logger.log_every(data_loader, print_freq, header, logger=logger):

        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        with torch.cuda.amp.autocast(enabled=args.amp):
            if need_tgt_for_training:
                outputs = model(samples, targets)
            else:
                outputs = model(samples)
        
            # loss_dict = criterion(outputs, targets)
            loss_dict = criterion(outputs, targets, model, 
                                                  module_type=args.module_type, expert_embeddings=None, 
                                                  batch_size=targets.__len__())
            weight_dict = criterion.weight_dict
            losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                      for k, v in loss_dict_reduced.items()}
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())

        loss_value = losses_reduced_scaled.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)


        # amp backward function
        if args.amp:
            optimizer.zero_grad()
            scaler.scale(losses).backward()
            if max_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            # original backward function
            optimizer.zero_grad()
            losses.backward()
            if max_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            optimizer.step()

        if args.onecyclelr:
            lr_scheduler.step()
        if args.use_ema:
            if epoch >= args.ema_epoch:
                ema_m.update(model)

        metric_logger.update(loss=loss_value, **loss_dict_reduced_scaled, **loss_dict_reduced_unscaled)
        if 'class_error' in loss_dict_reduced:
            metric_logger.update(class_error=loss_dict_reduced['class_error'])
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

        _cnt += 1
        if args.debug:
            if _cnt % 15 == 0:
                print("BREAK!"*5)
                break

    if getattr(criterion, 'loss_weight_decay', False):
        criterion.loss_weight_decay(epoch=epoch)
    if getattr(criterion, 'tuning_matching', False):
        criterion.tuning_matching(epoch)


    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    resstat = {k: meter.global_avg for k, meter in metric_logger.meters.items() if meter.count > 0}
    if getattr(criterion, 'loss_weight_decay', False):
        resstat.update({f'weight_{k}': v for k,v in criterion.weight_dict.items()})
    return resstat


def train_one_epoch_teaching_standard(model_student: torch.nn.Module, model_teacher: torch.nn.Module, criterion_student: torch.nn.Module, criterion_teacher: torch.nn.Module,
                    target_loader: Iterable, optimizer_student: torch.optim.Optimizer,
                    device: torch.device, epoch: int, threshold: int, max_norm: float = 0, 
                    wo_class_error=False, lr_scheduler_student=None, args=None, logger=None, ema_m=None):
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)

    try:
        need_tgt_for_training = args.use_dn
    except:
        need_tgt_for_training = False

    model_student.train()
    criterion_student.train()
    
    target_fetcher = DataPreFetcher(target_loader, device=device)
    target_images, target_masks, annotations = target_fetcher.next()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    if not wo_class_error:
        metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Epoch teacher: [{}]'.format(epoch)
    model_teacher.train()
    criterion_teacher.train()
    print_freq = 10
    alpha = 0.9
    top_k=5
    confidences = []
    save_dir = "saved_images"
    os.makedirs(save_dir, exist_ok=True)
    

    _cnt = 0
    for i in metric_logger.log_every(target_loader, print_freq, header, logger=logger):
        
        target_teacher_images, target_student_images = target_images[0], target_images[1]
    
        
        # Save target_teacher_images
        # for idx, image in enumerate(target_teacher_images):
        #     save_path = os.path.join(save_dir, f"target_teacher_image_{_cnt}.png")
        #     save_image(image, save_path)
        
        # # Save target_student_images
        # for idx, image in enumerate(target_student_images):
        #     save_path = os.path.join(save_dir, f"target_student_image_{_cnt}.png")
        #     save_image(image, save_path)

        with torch.cuda.amp.autocast(enabled=args.amp):
            with torch.no_grad():
                outputs_teacher = model_teacher(target_teacher_images)
                
                print("Dynamic threshold for standard teacher epoch ", epoch, " : ", threshold)
                pseudo_labels, highest_conf = create_pseudo_labels(outputs_teacher, threshold,  iou_threshold=0.25, top_k=top_k)
                confidences.append(highest_conf)
                

            ######################
            # TAKE ADVICE FROM EXPERTS
            pseudo_labels = [{**label, 'image_id': annotation['image_id'].item()} for label, annotation in zip(pseudo_labels, annotations) if 'image_id' in annotation]
            
            vitdino_embed_path = f'{args.expert_embed_dir}/vitdino/{args.source_dataset}/results_{args.target_dataset}/embeddings_save.npy'
            vitb16_clip_embed_path = f'{args.expert_embed_dir}/vitb16_clip/{args.source_dataset}/results_{args.target_dataset}/embeddings_save.npy'
            vitb16_imagenet_embed_path = f'{args.expert_embed_dir}/vitb16_imgnet/{args.source_dataset}/results_{args.target_dataset}/embeddings_save.npy'
            
            expert_embeddings_paths = [vitdino_embed_path, vitb16_clip_embed_path, vitb16_imagenet_embed_path]

            ####### Id alignment done for sim10k only #######
            if "sim10k" in vitdino_embed_path and "sim10k" in vitb16_clip_embed_path \
            and "sim10k" in vitb16_imagenet_embed_path:
                image_ids = [annotation['image_id'].item() - 500 for annotation in annotations]
            else:
                image_ids = [annotation['image_id'].item() for annotation in annotations]
            #################################################

            image_ids = torch.tensor(image_ids).cuda(device=device)
            
            outputs_student = model_student(target_student_images, take_expert_advice=True, 
                                            pseudo_labels=pseudo_labels, 
                                            expert_embeddings_paths=expert_embeddings_paths,
                                            image_ids=image_ids)
            
            loss_dict_student = criterion_student(outputs_student, pseudo_labels, model_student, 
                                                  module_type=args.module_type)

            # loss_dict_student = criterion_student(outputs_student, pseudo_labels, model_student, 
            #                                       module_type=args.module_type, expert_embeddings=None, 
            #                                       batch_size=target_student_images.shape[0])

            weight_dict_student = criterion_student.weight_dict
            losses_student = sum(loss_dict_student[k] * weight_dict_student[k] for k in loss_dict_student.keys() if k in weight_dict_student)

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced_student = utils.reduce_dict(loss_dict_student)
 
        loss_dict_reduced_unscaled_student = {f'{k}_unscaled': v
                                      for k, v in loss_dict_reduced_student.items()}

        loss_dict_reduced_scaled_student = {k: v * weight_dict_student[k]
                                    for k, v in loss_dict_reduced_student.items() if k in weight_dict_student}

        losses_reduced_scaled_student = sum(loss_dict_reduced_scaled_student.values())
        

        loss_value_student = losses_reduced_scaled_student.item()
        
        
        target_images, target_masks, annotations = target_fetcher.next()
        
        if not math.isfinite(loss_value_student):
            print("Student Loss is {}, stopping training".format(loss_value_student))
            print(loss_dict_reduced_student)
            sys.exit(1)
        

        
        if args.amp:
            optimizer_student.zero_grad()

            scaler.scale(losses_student).backward()
        
            if max_norm > 0:
                scaler.unscale_(optimizer_student)
                torch.nn.utils.clip_grad_norm_(model_student.parameters(), max_norm)
            scaler.step(optimizer_student)
            scaler.update()
        else:
            # original backward function
            optimizer_student.zero_grad()
            losses_student.backward()
            if max_norm > 0:
                torch.nn.utils.clip_grad_norm_(model_student.parameters(), max_norm)
            optimizer_student.step()

        if args.onecyclelr:
            lr_scheduler_student.step()
        if args.use_ema:
            if epoch >= args.ema_epoch:
                ema_m.update(model_student)

        metric_logger.update(loss=loss_value_student, **loss_dict_reduced_scaled_student, **loss_dict_reduced_unscaled_student)
        if 'class_error' in loss_dict_reduced_student:
            metric_logger.update(class_error=loss_dict_reduced_student['class_error'])    
        metric_logger.update(lr=optimizer_student.param_groups[0]["lr"])
        
        _cnt += 1
        if args.debug:
            if _cnt % 15 == 0:
                print("BREAK!"*5)
                break

    if getattr(criterion_student, 'loss_weight_decay', False):
        criterion_student.loss_weight_decay(epoch=epoch)
    if getattr(criterion_student, 'tuning_matching', False):
        criterion_student.tuning_matching(epoch)



    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats for teaching:", metric_logger)
    resstat_student = {k: meter.global_avg for k, meter in metric_logger.meters.items() if meter.count > 0}        
    

    if getattr(criterion_student, 'loss_weight_decay', False):
        resstat_student.update({f'weight_{k}': v for k,v in criterion_student.weight_dict.items()})

    # Teacher update through EMA
    with torch.no_grad():
        state_dict, student_state_dict = model_teacher.state_dict(), model_student.state_dict()
        for key, value in state_dict.items():
            state_dict[key] = alpha * value + (1 - alpha) * student_state_dict[key].detach()
        model_teacher.load_state_dict(state_dict)
    

    
    print("At the end of training epoch...")
    
    return resstat_student, confidences

def train_one_epoch_teaching_new(model_student: torch.nn.Module, model_teacher: torch.nn.Module, criterion_student: torch.nn.Module, criterion_teacher: torch.nn.Module,
                    target_loader: Iterable, optimizer_student: torch.optim.Optimizer,
                    device: torch.device, epoch: int, threshold: int, max_norm: float = 0, 
                    wo_class_error=False, lr_scheduler_student=None, args=None, logger=None, ema_m=None):
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)

    try:
        need_tgt_for_training = args.use_dn
    except:
        need_tgt_for_training = False

    model_student.train()
    criterion_student.train()
    
    target_fetcher = DataPreFetcher(target_loader, device=device)
    target_images, target_masks, annotations = target_fetcher.next()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    if not wo_class_error:
        metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Epoch teacher: [{}]'.format(epoch)
    model_teacher.train()
    criterion_teacher.train()
    print_freq = 10
    confidences = []
    save_dir = "saved_images"
    os.makedirs(save_dir, exist_ok=True)
    

    _cnt = 0
    for i in metric_logger.log_every(target_loader, print_freq, header, logger=logger):
        
        target_teacher_images, target_student_images = target_images[0], target_images[1]
    
        
        # Save target_teacher_images
        # for idx, image in enumerate(target_teacher_images):
        #     save_path = os.path.join(save_dir, f"target_teacher_image_{_cnt}.png")
        #     save_image(image, save_path)
        
        # # Save target_student_images
        # for idx, image in enumerate(target_student_images):
        #     save_path = os.path.join(save_dir, f"target_student_image_{_cnt}.png")
        #     save_image(image, save_path)

        with torch.cuda.amp.autocast(enabled=args.amp):
            with torch.no_grad():
                outputs_teacher = model_teacher(target_teacher_images)
                
                print("Dynamic threshold for standard teacher epoch ", epoch, " : ", threshold)
                pseudo_labels, highest_conf = create_pseudo_labels(outputs_teacher, threshold)
                confidences.append(highest_conf)
                

            ######################
            # TAKE ADVICE FROM EXPERTS
            pseudo_labels = [{**label, 'image_id': annotation['image_id'].item()} for label, annotation in zip(pseudo_labels, annotations) if 'image_id' in annotation]
            
            vitdino_embed_path = f'{args.expert_embed_dir}/vitdino/{args.source_dataset}/results_{args.target_dataset}/embeddings_save.npy'
            vitb16_clip_embed_path = f'{args.expert_embed_dir}/vitb16_clip/{args.source_dataset}/results_{args.target_dataset}/embeddings_save.npy'
            vitb16_imagenet_embed_path = f'{args.expert_embed_dir}/vitb16_imgnet/{args.source_dataset}/results_{args.target_dataset}/embeddings_save.npy'
            
            expert_embeddings_paths = [vitdino_embed_path, vitb16_clip_embed_path, vitb16_imagenet_embed_path]

            ####### Id alignment done for sim10k only #######
            if "sim10k" in vitdino_embed_path and "sim10k" in vitb16_clip_embed_path \
            and "sim10k" in vitb16_imagenet_embed_path:
                image_ids = [annotation['image_id'].item() - 500 for annotation in annotations]
            else:
                image_ids = [annotation['image_id'].item() for annotation in annotations]
            #################################################

            image_ids = torch.tensor(image_ids).cuda(device=device)
            
            outputs_student = model_student(target_student_images, take_expert_advice=True, 
                                            pseudo_labels=pseudo_labels, 
                                            expert_embeddings_paths=expert_embeddings_paths,
                                            image_ids=image_ids)
            
            loss_dict_student = criterion_student(outputs_student, pseudo_labels, model_student, 
                                                  module_type=args.module_type)

            # loss_dict_student = criterion_student(outputs_student, pseudo_labels, model_student, 
            #                                       module_type=args.module_type, expert_embeddings=None, 
            #                                       batch_size=target_student_images.shape[0])

            weight_dict_student = criterion_student.weight_dict
            losses_student = sum(loss_dict_student[k] * weight_dict_student[k] for k in loss_dict_student.keys() if k in weight_dict_student)

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced_student = utils.reduce_dict(loss_dict_student)
 
        loss_dict_reduced_unscaled_student = {f'{k}_unscaled': v
                                      for k, v in loss_dict_reduced_student.items()}

        loss_dict_reduced_scaled_student = {k: v * weight_dict_student[k]
                                    for k, v in loss_dict_reduced_student.items() if k in weight_dict_student}

        losses_reduced_scaled_student = sum(loss_dict_reduced_scaled_student.values())
        

        loss_value_student = losses_reduced_scaled_student.item()
        
        
        target_images, target_masks, annotations = target_fetcher.next()
        
        if not math.isfinite(loss_value_student):
            print("Student Loss is {}, stopping training".format(loss_value_student))
            print(loss_dict_reduced_student)
            sys.exit(1)
        

        
        if args.amp:
            optimizer_student.zero_grad()

            scaler.scale(losses_student).backward()
        
            if max_norm > 0:
                scaler.unscale_(optimizer_student)
                torch.nn.utils.clip_grad_norm_(model_student.parameters(), max_norm)
            scaler.step(optimizer_student)
            scaler.update()
        else:
            # original backward function
            optimizer_student.zero_grad()
            losses_student.backward()
            if max_norm > 0:
                torch.nn.utils.clip_grad_norm_(model_student.parameters(), max_norm)
            optimizer_student.step()

        if args.onecyclelr:
            lr_scheduler_student.step()
        if args.use_ema:
            if epoch >= args.ema_epoch:
                ema_m.update(model_student)

        metric_logger.update(loss=loss_value_student, **loss_dict_reduced_scaled_student, **loss_dict_reduced_unscaled_student)
        if 'class_error' in loss_dict_reduced_student:
            metric_logger.update(class_error=loss_dict_reduced_student['class_error'])    
        metric_logger.update(lr=optimizer_student.param_groups[0]["lr"])
        
        _cnt += 1
        if args.debug:
            if _cnt % 15 == 0:
                print("BREAK!"*5)
                break

    if getattr(criterion_student, 'loss_weight_decay', False):
        criterion_student.loss_weight_decay(epoch=epoch)
    if getattr(criterion_student, 'tuning_matching', False):
        criterion_student.tuning_matching(epoch)



    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats for teaching:", metric_logger)
    resstat_student = {k: meter.global_avg for k, meter in metric_logger.meters.items() if meter.count > 0}        
    

    if getattr(criterion_student, 'loss_weight_decay', False):
        resstat_student.update({f'weight_{k}': v for k,v in criterion_student.weight_dict.items()})

    # Teacher update through EMA
    with torch.no_grad():
        state_dict, student_state_dict = model_teacher.state_dict(), model_student.state_dict()
        for key, value in state_dict.items():
            state_dict[key] = 0.9 * value + (1 - 0.9) * student_state_dict[key].detach()
        model_teacher.load_state_dict(state_dict)
    

    
    print("At the end of training epoch...")
    
    return resstat_student, confidences


@torch.no_grad()
def evaluate(model, criterion, postprocessors, data_loader, base_ds, device, output_dir, wo_class_error=False, args=None, logger=None):
    try:
        need_tgt_for_training = args.use_dn
    except:
        need_tgt_for_training = False

    model.eval()
    criterion.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    if not wo_class_error:
        metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Test:'

    iou_types = tuple(k for k in ('segm', 'bbox') if k in postprocessors.keys())
    useCats = True
    try:
        useCats = args.useCats
    except:
        useCats = True
    if not useCats:
        print("useCats: {} !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!".format(useCats))
    coco_evaluator = CocoEvaluator(base_ds, iou_types, useCats=useCats)
    # coco_evaluator.coco_eval[iou_types[0]].params.iouThrs = [0, 0.1, 0.5, 0.75]

    panoptic_evaluator = None
    if 'panoptic' in postprocessors.keys():
        panoptic_evaluator = PanopticEvaluator(
            data_loader.dataset.ann_file,
            data_loader.dataset.ann_folder,
            output_dir=os.path.join(output_dir, "panoptic_eval"),
        )

    _cnt = 0
    output_state_dict = {} # for debug only
    for samples, targets in metric_logger.log_every(data_loader, 10, header, logger=logger):
        samples = samples.to(device)
        targets = [{k: to_device(v, device) for k, v in t.items()} for t in targets]

        with torch.cuda.amp.autocast(enabled=args.amp):
            if need_tgt_for_training:
                outputs = model(samples, targets)
            else:
                outputs = model(samples)
            # outputs = model(samples)

            loss_dict = criterion(outputs, targets, model, module_type="student_with_no_expert", expert_embeddings=None)
        weight_dict = criterion.weight_dict

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                      for k, v in loss_dict_reduced.items()}
        metric_logger.update(loss=sum(loss_dict_reduced_scaled.values()),
                             **loss_dict_reduced_scaled,
                             **loss_dict_reduced_unscaled)
        if 'class_error' in loss_dict_reduced:
            metric_logger.update(class_error=loss_dict_reduced['class_error'])

        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
        results = postprocessors['bbox'](outputs, orig_target_sizes)
        # [scores: [100], labels: [100], boxes: [100, 4]] x B
        if 'segm' in postprocessors.keys():
            target_sizes = torch.stack([t["size"] for t in targets], dim=0)
            results = postprocessors['segm'](results, outputs, orig_target_sizes, target_sizes)
        res = {target['image_id'].item(): output for target, output in zip(targets, results)}
        
        if coco_evaluator is not None:
            coco_evaluator.update(res)

        if panoptic_evaluator is not None:
            res_pano = postprocessors["panoptic"](outputs, target_sizes, orig_target_sizes)
            for i, target in enumerate(targets):
                image_id = target["image_id"].item()
                file_name = f"{image_id:012d}.png"
                res_pano[i]["image_id"] = image_id
                res_pano[i]["file_name"] = file_name

            panoptic_evaluator.update(res_pano)
        
        if args.save_results:
            # res_score = outputs['res_score']
            # res_label = outputs['res_label']
            # res_bbox = outputs['res_bbox']
            # res_idx = outputs['res_idx']
            # import ipdb; ipdb.set_trace()

            for i, (tgt, res, outbbox) in enumerate(zip(targets, results, outputs['pred_boxes'])):
                """
                pred vars:
                    K: number of bbox pred
                    score: Tensor(K),
                    label: list(len: K),
                    bbox: Tensor(K, 4)
                    idx: list(len: K)
                tgt: dict.

                """
                # compare gt and res (after postprocess)
                gt_bbox = tgt['boxes']
                gt_label = tgt['labels']
                gt_info = torch.cat((gt_bbox, gt_label.unsqueeze(-1)), 1)
                
                # img_h, img_w = tgt['orig_size'].unbind()
                # scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=0)
                # _res_bbox = res['boxes'] / scale_fct
                _res_bbox = outbbox
                _res_prob = res['scores']
                _res_label = res['labels']
                res_info = torch.cat((_res_bbox, _res_prob.unsqueeze(-1), _res_label.unsqueeze(-1)), 1)

                if 'gt_info' not in output_state_dict:
                    output_state_dict['gt_info'] = []
                output_state_dict['gt_info'].append(gt_info.cpu())

                if 'res_info' not in output_state_dict:
                    output_state_dict['res_info'] = []
                output_state_dict['res_info'].append(res_info.cpu())

            # # for debug only
            # import random
            # if random.random() > 0.7:
            #     print("Now let's break")
            #     break

        _cnt += 1
        if args.debug:
            if _cnt % 15 == 0:
                print("BREAK!"*5)
                break

    if args.save_results:
        import os.path as osp
        
        # output_state_dict['gt_info'] = torch.cat(output_state_dict['gt_info'])
        # output_state_dict['res_info'] = torch.cat(output_state_dict['res_info'])
        savepath = osp.join(args.output_dir, 'results-{}.pkl'.format(utils.get_rank()))
        print("Saving res to {}".format(savepath))
        torch.save(output_state_dict, savepath)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    if coco_evaluator is not None:
        coco_evaluator.synchronize_between_processes()
    if panoptic_evaluator is not None:
        panoptic_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    if coco_evaluator is not None:
        coco_evaluator.accumulate()
        coco_evaluator.summarize()
        
    panoptic_res = None
    if panoptic_evaluator is not None:
        panoptic_res = panoptic_evaluator.summarize()
    stats = {k: meter.global_avg for k, meter in metric_logger.meters.items() if meter.count > 0}
    if coco_evaluator is not None:
        if 'bbox' in postprocessors.keys():
            stats['coco_eval_bbox'] = coco_evaluator.coco_eval['bbox'].stats.tolist()
        if 'segm' in postprocessors.keys():
            stats['coco_eval_masks'] = coco_evaluator.coco_eval['segm'].stats.tolist()
    if panoptic_res is not None:
        stats['PQ_all'] = panoptic_res["All"]
        stats['PQ_th'] = panoptic_res["Things"]
        stats['PQ_st'] = panoptic_res["Stuff"]

    # import ipdb; ipdb.set_trace()

    return stats, coco_evaluator


@torch.no_grad()
def evaluate_teacher(model, criterion, postprocessors, data_loader, base_ds, device, output_dir, wo_class_error=False, args=None, logger=None):
    try:
        need_tgt_for_training = args.use_dn
    except:
        need_tgt_for_training = False

    model.eval()
    criterion.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    if not wo_class_error:
        metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Test:'

    iou_types = tuple(k for k in ('segm', 'bbox') if k in postprocessors.keys())
    useCats = True
    try:
        useCats = args.useCats
    except:
        useCats = True
    if not useCats:
        print("useCats: {} !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!".format(useCats))
    coco_evaluator = CocoEvaluator(base_ds, iou_types, useCats=useCats)
    # coco_evaluator.coco_eval[iou_types[0]].params.iouThrs = [0, 0.1, 0.5, 0.75]

    panoptic_evaluator = None
    if 'panoptic' in postprocessors.keys():
        panoptic_evaluator = PanopticEvaluator(
            data_loader.dataset.ann_file,
            data_loader.dataset.ann_folder,
            output_dir=os.path.join(output_dir, "panoptic_eval"),
        )

    _cnt = 0
    output_state_dict = {} # for debug only
    for samples, targets in metric_logger.log_every(data_loader, 10, header, logger=logger):
        samples = samples.to(device)
        # import ipdb; ipdb.set_trace()
        # targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        targets = [{k: to_device(v, device) for k, v in t.items()} for t in targets]

        with torch.cuda.amp.autocast(enabled=args.amp):
            if need_tgt_for_training:
                outputs = model(samples, targets)
            else:
                outputs = model(samples)
            # outputs = model(samples)

            loss_dict = criterion(outputs, targets)
        weight_dict = criterion.weight_dict

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                      for k, v in loss_dict_reduced.items()}
        metric_logger.update(loss=sum(loss_dict_reduced_scaled.values()),
                             **loss_dict_reduced_scaled,
                             **loss_dict_reduced_unscaled)
        if 'class_error' in loss_dict_reduced:
            metric_logger.update(class_error=loss_dict_reduced['class_error'])

        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
        results = postprocessors['bbox'](outputs, orig_target_sizes)
        # [scores: [100], labels: [100], boxes: [100, 4]] x B
        if 'segm' in postprocessors.keys():
            target_sizes = torch.stack([t["size"] for t in targets], dim=0)
            results = postprocessors['segm'](results, outputs, orig_target_sizes, target_sizes)
        res = {target['image_id'].item(): output for target, output in zip(targets, results)}
        # import ipdb; ipdb.set_trace()
        if coco_evaluator is not None:
            coco_evaluator.update(res)

        if panoptic_evaluator is not None:
            res_pano = postprocessors["panoptic"](outputs, target_sizes, orig_target_sizes)
            for i, target in enumerate(targets):
                image_id = target["image_id"].item()
                file_name = f"{image_id:012d}.png"
                res_pano[i]["image_id"] = image_id
                res_pano[i]["file_name"] = file_name

            panoptic_evaluator.update(res_pano)
        
        if args.save_results:
            # res_score = outputs['res_score']
            # res_label = outputs['res_label']
            # res_bbox = outputs['res_bbox']
            # res_idx = outputs['res_idx']
            # import ipdb; ipdb.set_trace()

            for i, (tgt, res, outbbox) in enumerate(zip(targets, results, outputs['pred_boxes'])):
                """
                pred vars:
                    K: number of bbox pred
                    score: Tensor(K),
                    label: list(len: K),
                    bbox: Tensor(K, 4)
                    idx: list(len: K)
                tgt: dict.

                """
                # compare gt and res (after postprocess)
                gt_bbox = tgt['boxes']
                gt_label = tgt['labels']
                gt_info = torch.cat((gt_bbox, gt_label.unsqueeze(-1)), 1)
                
                # img_h, img_w = tgt['orig_size'].unbind()
                # scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=0)
                # _res_bbox = res['boxes'] / scale_fct
                _res_bbox = outbbox
                _res_prob = res['scores']
                _res_label = res['labels']
                res_info = torch.cat((_res_bbox, _res_prob.unsqueeze(-1), _res_label.unsqueeze(-1)), 1)
                # import ipdb;ipdb.set_trace()

                if 'gt_info' not in output_state_dict:
                    output_state_dict['gt_info'] = []
                output_state_dict['gt_info'].append(gt_info.cpu())

                if 'res_info' not in output_state_dict:
                    output_state_dict['res_info'] = []
                output_state_dict['res_info'].append(res_info.cpu())

            # # for debug only
            # import random
            # if random.random() > 0.7:
            #     print("Now let's break")
            #     break

        _cnt += 1
        if args.debug:
            if _cnt % 15 == 0:
                print("BREAK!"*5)
                break

    if args.save_results:
        import os.path as osp
        
        # output_state_dict['gt_info'] = torch.cat(output_state_dict['gt_info'])
        # output_state_dict['res_info'] = torch.cat(output_state_dict['res_info'])
        savepath = osp.join(args.output_dir, 'results-{}.pkl'.format(utils.get_rank()))
        print("Saving teacher res to {}".format(savepath))
        torch.save(output_state_dict, savepath)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged teacher stats:", metric_logger)
    if coco_evaluator is not None:
        coco_evaluator.synchronize_between_processes()
    if panoptic_evaluator is not None:
        panoptic_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    if coco_evaluator is not None:
        coco_evaluator.accumulate()
        coco_evaluator.summarize()
        
    panoptic_res = None
    if panoptic_evaluator is not None:
        panoptic_res = panoptic_evaluator.summarize()
    stats = {k: meter.global_avg for k, meter in metric_logger.meters.items() if meter.count > 0}
    if coco_evaluator is not None:
        if 'bbox' in postprocessors.keys():
            stats['coco_eval_bbox'] = coco_evaluator.coco_eval['bbox'].stats.tolist()
        if 'segm' in postprocessors.keys():
            stats['coco_eval_masks'] = coco_evaluator.coco_eval['segm'].stats.tolist()
    if panoptic_res is not None:
        stats['PQ_all'] = panoptic_res["All"]
        stats['PQ_th'] = panoptic_res["Things"]
        stats['PQ_st'] = panoptic_res["Stuff"]

    # import ipdb; ipdb.set_trace()

    return stats, coco_evaluator

@torch.no_grad()
def evaluate_student(model, criterion, postprocessors, data_loader, base_ds, device, output_dir, wo_class_error=False, args=None, logger=None):
    try:
        need_tgt_for_training = args.use_dn
    except:
        need_tgt_for_training = False

    model.eval()
    criterion.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    if not wo_class_error:
        metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Test:'

    iou_types = tuple(k for k in ('segm', 'bbox') if k in postprocessors.keys())
    useCats = True
    try:
        useCats = args.useCats
    except:
        useCats = True
    if not useCats:
        print("useCats: {} !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!".format(useCats))
    coco_evaluator = CocoEvaluator(base_ds, iou_types, useCats=useCats)
    # coco_evaluator.coco_eval[iou_types[0]].params.iouThrs = [0, 0.1, 0.5, 0.75]

    panoptic_evaluator = None
    if 'panoptic' in postprocessors.keys():
        panoptic_evaluator = PanopticEvaluator(
            data_loader.dataset.ann_file,
            data_loader.dataset.ann_folder,
            output_dir=os.path.join(output_dir, "panoptic_eval"),
        )

    _cnt = 0
    output_state_dict = {} # for debug only
    # import ipdb; ipdb.set_trace()
    for samples, targets in metric_logger.log_every(data_loader, 10, header, logger=logger):
        samples = samples.to(device)
        # import ipdb; ipdb.set_trace()
        # targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        targets = [{k: to_device(v, device) for k, v in t.items()} for t in targets]

        with torch.cuda.amp.autocast(enabled=args.amp):
            if need_tgt_for_training:
                outputs = model(samples, targets)
            else:
                outputs = model(samples)
            # outputs = model(samples)

            loss_dict = criterion(outputs, targets)
        weight_dict = criterion.weight_dict

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                      for k, v in loss_dict_reduced.items()}
        metric_logger.update(loss=sum(loss_dict_reduced_scaled.values()),
                             **loss_dict_reduced_scaled,
                             **loss_dict_reduced_unscaled)
        if 'class_error' in loss_dict_reduced:
            metric_logger.update(class_error=loss_dict_reduced['class_error'])

        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
        results = postprocessors['bbox'](outputs, orig_target_sizes)
        # [scores: [100], labels: [100], boxes: [100, 4]] x B
        if 'segm' in postprocessors.keys():
            target_sizes = torch.stack([t["size"] for t in targets], dim=0)
            results = postprocessors['segm'](results, outputs, orig_target_sizes, target_sizes)
        res = {target['image_id'].item(): output for target, output in zip(targets, results)}
        # import ipdb; ipdb.set_trace()
        if coco_evaluator is not None:
            coco_evaluator.update(res)

        if panoptic_evaluator is not None:
            res_pano = postprocessors["panoptic"](outputs, target_sizes, orig_target_sizes)
            for i, target in enumerate(targets):
                image_id = target["image_id"].item()
                file_name = f"{image_id:012d}.png"
                res_pano[i]["image_id"] = image_id
                res_pano[i]["file_name"] = file_name

            panoptic_evaluator.update(res_pano)
        
        if args.save_results:
            # res_score = outputs['res_score']
            # res_label = outputs['res_label']
            # res_bbox = outputs['res_bbox']
            # res_idx = outputs['res_idx']
            # import ipdb; ipdb.set_trace()

            for i, (tgt, res, outbbox) in enumerate(zip(targets, results, outputs['pred_boxes'])):
                """
                pred vars:
                    K: number of bbox pred
                    score: Tensor(K),
                    label: list(len: K),
                    bbox: Tensor(K, 4)
                    idx: list(len: K)
                tgt: dict.

                """
                # compare gt and res (after postprocess)
                gt_bbox = tgt['boxes']
                gt_label = tgt['labels']
                gt_info = torch.cat((gt_bbox, gt_label.unsqueeze(-1)), 1)
                
                # img_h, img_w = tgt['orig_size'].unbind()
                # scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=0)
                # _res_bbox = res['boxes'] / scale_fct
                _res_bbox = outbbox
                _res_prob = res['scores']
                _res_label = res['labels']
                res_info = torch.cat((_res_bbox, _res_prob.unsqueeze(-1), _res_label.unsqueeze(-1)), 1)
                # import ipdb;ipdb.set_trace()

                if 'gt_info' not in output_state_dict:
                    output_state_dict['gt_info'] = []
                output_state_dict['gt_info'].append(gt_info.cpu())

                if 'res_info' not in output_state_dict:
                    output_state_dict['res_info'] = []
                output_state_dict['res_info'].append(res_info.cpu())

            # # for debug only
            # import random
            # if random.random() > 0.7:
            #     print("Now let's break")
            #     break

        _cnt += 1
        if args.debug:
            if _cnt % 15 == 0:
                print("BREAK!"*5)
                break

    if args.save_results:
        import os.path as osp
        
        # output_state_dict['gt_info'] = torch.cat(output_state_dict['gt_info'])
        # output_state_dict['res_info'] = torch.cat(output_state_dict['res_info'])
        savepath = osp.join(args.output_dir, 'results-{}.pkl'.format(utils.get_rank()))
        print("Saving student res to {}".format(savepath))
        torch.save(output_state_dict, savepath)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged Student stats:", metric_logger)
    if coco_evaluator is not None:
        coco_evaluator.synchronize_between_processes()
    if panoptic_evaluator is not None:
        panoptic_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    if coco_evaluator is not None:
        coco_evaluator.accumulate()
        coco_evaluator.summarize()
        
    panoptic_res = None
    if panoptic_evaluator is not None:
        panoptic_res = panoptic_evaluator.summarize()
    stats = {k: meter.global_avg for k, meter in metric_logger.meters.items() if meter.count > 0}
    if coco_evaluator is not None:
        if 'bbox' in postprocessors.keys():
            stats['coco_eval_bbox'] = coco_evaluator.coco_eval['bbox'].stats.tolist()
        if 'segm' in postprocessors.keys():
            stats['coco_eval_masks'] = coco_evaluator.coco_eval['segm'].stats.tolist()
    if panoptic_res is not None:
        stats['PQ_all'] = panoptic_res["All"]
        stats['PQ_th'] = panoptic_res["Things"]
        stats['PQ_st'] = panoptic_res["Stuff"]

    # import ipdb; ipdb.set_trace()

    return stats, coco_evaluator



@torch.no_grad()
def test(model, criterion, postprocessors, data_loader, base_ds, device, output_dir, wo_class_error=False, args=None, logger=None):
    model.eval()
    criterion.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    # if not wo_class_error:
    #     metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Test:'

    iou_types = tuple(k for k in ('segm', 'bbox') if k in postprocessors.keys())
    # coco_evaluator = CocoEvaluator(base_ds, iou_types)
    # coco_evaluator.coco_eval[iou_types[0]].params.iouThrs = [0, 0.1, 0.5, 0.75]

    panoptic_evaluator = None
    if 'panoptic' in postprocessors.keys():
        panoptic_evaluator = PanopticEvaluator(
            data_loader.dataset.ann_file,
            data_loader.dataset.ann_folder,
            output_dir=os.path.join(output_dir, "panoptic_eval"),
        )

    final_res = []
    for samples, targets in metric_logger.log_every(data_loader, 10, header, logger=logger):
        samples = samples.to(device)
        # import ipdb; ipdb.set_trace()
        # targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        targets = [{k: to_device(v, device) for k, v in t.items()} for t in targets]

        outputs = model(samples)
        # loss_dict = criterion(outputs, targets)
        # weight_dict = criterion.weight_dict

        # # reduce losses over all GPUs for logging purposes
        # loss_dict_reduced = utils.reduce_dict(loss_dict)
        # loss_dict_reduced_scaled = {k: v * weight_dict[k]
        #                             for k, v in loss_dict_reduced.items() if k in weight_dict}
        # loss_dict_reduced_unscaled = {f'{k}_unscaled': v
        #                               for k, v in loss_dict_reduced.items()}
        # metric_logger.update(loss=sum(loss_dict_reduced_scaled.values()),
        #                      **loss_dict_reduced_scaled,
        #                      **loss_dict_reduced_unscaled)
        # if 'class_error' in loss_dict_reduced:
        #     metric_logger.update(class_error=loss_dict_reduced['class_error'])

        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
        results = postprocessors['bbox'](outputs, orig_target_sizes, not_to_xyxy=True)
        # [scores: [100], labels: [100], boxes: [100, 4]] x B
        if 'segm' in postprocessors.keys():
            target_sizes = torch.stack([t["size"] for t in targets], dim=0)
            results = postprocessors['segm'](results, outputs, orig_target_sizes, target_sizes)
        res = {target['image_id'].item(): output for target, output in zip(targets, results)}
        for image_id, outputs in res.items():
            _scores = outputs['scores'].tolist()
            _labels = outputs['labels'].tolist()
            _boxes = outputs['boxes'].tolist()
            for s, l, b in zip(_scores, _labels, _boxes):
                assert isinstance(l, int)
                itemdict = {
                        "image_id": int(image_id), 
                        "category_id": l, 
                        "bbox": b, 
                        "score": s,
                        }
                final_res.append(itemdict)

    if args.output_dir:
        import json
        with open(args.output_dir + f'/results{args.rank}.json', 'w') as f:
            json.dump(final_res, f)        
        
    # gather the stats from all processes
    # metric_logger.synchronize_between_processes()
    # print("Averaged stats:", metric_logger)
    # if coco_evaluator is not None:
    #     coco_evaluator.synchronize_between_processes()
    # if panoptic_evaluator is not None:
    #     panoptic_evaluator.synchronize_between_processes()

    # # accumulate predictions from all images
    # if coco_evaluator is not None:
    #     coco_evaluator.accumulate()
    #     coco_evaluator.summarize()
        
    # panoptic_res = None
    # if panoptic_evaluator is not None:
    #     panoptic_res = panoptic_evaluator.summarize()
    # stats = {k: meter.global_avg for k, meter in metric_logger.meters.items() if meter.count > 0}
    # if coco_evaluator is not None:
    #     if 'bbox' in postprocessors.keys():
    #         stats['coco_eval_bbox'] = coco_evaluator.coco_eval['bbox'].stats.tolist()
    #     if 'segm' in postprocessors.keys():
    #         stats['coco_eval_masks'] = coco_evaluator.coco_eval['segm'].stats.tolist()
    # if panoptic_res is not None:
    #     stats['PQ_all'] = panoptic_res["All"]
    #     stats['PQ_th'] = panoptic_res["Things"]
    #     stats['PQ_st'] = panoptic_res["Stuff"]

    # import ipdb; ipdb.set_trace()

    return final_res
