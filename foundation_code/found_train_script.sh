# CUDA_VISIBLE_DEVICES=5 python ./code/train_any_foundation.py 
# --checkpoint_model_save <save_path> 
# --topk 40 --num_epochs 120 --num_workers 8 --batch_size 8 --layers_freeze 2 --model_type vitclip 
# --learning_rate 5e-5 --img_size 224 --class_num 22

# CUDA_VISIBLE_DEVICES=5 python ./code/train_any_foundation.py --checkpoint_model_save <save_path> --topk 40 --num_epochs 120 --num_workers 8 --batch_size 8 --layers_freeze 2 --model_type vitclip --learning_rate 5e-5 --img_size 224 --class_num 22