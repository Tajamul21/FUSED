## FIRST THINGS FIRST
Follow through the environment installation in the repo's root, and activate ```found``` environment.

# 1) SOURCE PRETRAINING  

There are two step **before** we go about training them. First is getting the crops from the source training FND model, and then preparing
the classification groundtruth file for the foundation models to be trained.  

These steps will be repeated for the ```TEST``` split as well.  

_**Step-1**_
_Getting the Crops from FND model._
For this you need to shift to ```FUSED/``` and run the file ```create_preds_multi_modal.py```.  
Don't forget to set appropriate paths in the file according to your desires.
```
python create_preds_multi_modal.py
```

_**Step-2**_
_Creating classification ground truth files for foundation models training.  _
Don't forget to set appropriate paths in the file according to your desires.  
It is recommended to set the ```output_path``` variable as the same directory as your ```source dataset```.
```
python create_annotations_for_foundation.py
```

### FOUNDATION SOURCE PRE-TRAINING
After following the above steps, to train a foundation model, you need to run the following command:  
```
CUDA_VISIBLE_DEVICES=5 python train_any_foundation.py \
--checkpoint_model_save /home/keerti_km/scratch/another-sfda-reproduced/SFDA-NeurIPS-24/foundation_weights/vitb16_clip/inhouse \
--topk 40 --num_epochs 120 --num_workers 8 --batch_size 8 --layers_freeze 2 --model_type vitb16_clip --learning_rate 5e-5 --img_size 224 --class_num 2
```
This training command is run for every foundation model. It is recommended to set ```--checkpoint_model_save``` path to 
``` ../foundation_weights/{model_type}/{dataset_source} ```
This would save a ```model_best.pt``` checkpoint for the foundation model in ```../foundation_weights``` in the concerned ```source``` dataset folder


### OBTAINING EMBEDDINGS FROM FOUNDATION MODELS FOR TARGET SET  
_**Step-1**_  
_Getting the crops from FND detector for Target Set:_
For this you need to shift to ```FUSED/``` and run the file ```create_preds_multi_modal.py```.  
Don't forget to set appropriate paths in the file according to your desires.
```
python create_preds_multi_modal.py
```

_**Step-2**_  
_Creating Ground truth for foundation models for Target Set:_
Don't forget to set appropriate paths in the file according to your desires.  
It is recommended to set the ```output_path``` variable as the same directory as your ```target dataset```.

```
python create_annotations_for_foundation.py
```

_**Step-3**_
(DON'T FOGET TO CREATE ```../foundation_weights``` DIRECTORY)
_Now finally, creating the foundation embeddings for target set:_  
```
CUDA_VISIBLE_DEVICES=4 python test.py --model_type vitb16_clip --dataset_source inhouse --dataset_target inbreast --class_num 2 --topk 20 --img_size 224 --layers_freeze 2
```
**NOTE:**  
- If # of classes is 1, then input 2 in ```--class_num``` argument, if # of classes is > 1, then keep the number same as it is.  
- These embeddings will be saved in ```../foundation_weights``` folder, same as where source trained models are saved.
