#!/bin/bash


CUDA_VISIBLE_DEVICES=6,7 python -m torch.distributed.launch --nproc_per_node=2 --master_port 11008 main.py \
--config_file /home/tajamul/scratch/SFDA-NeurIPS-24/FUSED/configs/config_cfg.py \
--source_dataset ddsm \
--target_dataset inbreast \
--coco_path /home/tajamul/scratch/DA/DATA/Coco_Data/BCD/ \
--pretrain_model_path /home/tajamul/scratch/SFDA-NeurIPS-24/FUSED/Source_only/DDSM/checkpoint0022.pth \
--output_dir /home/tajamul/scratch/DA/Checkpoints/Reproduce/ddsm/ddsm2inbreast \
--find_unused_params \
--find_unused_params --module_type expert --adapt --temp 1.0 --memory_bank_size 1 \
--adapt \
--init_threshold 0.06 \
--lambda_eckd 1.0 \
--conf_update_algo const_thresh \
--expert_embed_dir ../foundation_weights
        

# student_with_no_expert
# expert