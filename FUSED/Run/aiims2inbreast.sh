#!/bin/bash

CUDA_VISIBLE_DEVICES=6,7 python -m torch.distributed.launch --nproc_per_node=2 --master_port 11001 main.py \
--config_file /home/tajamul/scratch/DA/SFDA/target_directory/SFDA-NeurIPS-24/FUSED/configs/config_cfg.py \
--source_dataset inhouse \
--target_dataset inbreast \
--coco_path /home/tajamul/scratch/DA/DATA/Coco_Data/BCD/ \
--pretrain_model_path /home/tajamul/scratch/Reproduce_code/Source_only/Aiims/checkpoint0025.pth \
--output_dir /home/tajamul/scratch/DA/Checkpoints/Reproduce/Improvement/aiims2inbreaststandard_final_check \
--find_unused_params \
--find_unused_params --module_type student_with_no_expert --adapt --temp 1.0 --memory_bank_size 1 \
--adapt \
--init_threshold 0.06 \
--lambda_eckd 1.0 \
--conf_update_algo raise_abruptly \
--expert_embed_dir ../foundation_weights
 
