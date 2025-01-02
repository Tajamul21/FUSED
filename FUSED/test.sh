output_dir=$1
config=$2
coco_path=$3
checkpoint=$4
lambda_eckd=$5
temp=$6

python test.py \
  --output_dir $output_dir \
  -c $config --coco_path $coco_path  \
	--eval_test --resume $checkpoint \
	--lambda_eckd $lambda_eckd --memory_bank_size 0 --temp $temp \
	--options dn_scalar=100 embed_init_tgt=TRUE \
	dn_label_coef=1.0 dn_bbox_coef=1.0 use_ema=False \
	dn_box_noise_scale=1.0