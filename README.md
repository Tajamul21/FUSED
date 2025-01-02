# Foundational Models Guide Best: Source Free Domain Adaptive Breast Cancer Detection from Mammograms

# Abstract
We focus on source-free domain adaptive object detection (\sfdaod), when source data is unavailable during adaptation and the model must adapt on the unlabeled target domain. Majority of approaches for the problem employ a self-supervised approach using a student-teacher (\st) framework. We observe that the performance of a student model often degrades dramatically after \sfdaod, even worse than no domain adaptation. This happens due to teacher model collapsing because of source data bias and large domain shift between source and target domains. On the other hand, vision foundation models (\vfms) such as \clip, \dinov, \sam have demonstrated excellent zero shot performance for various downstream tasks. While trained with distinct objectives and exhibiting unique traits, these models can be unified effectively through multi-expert distillation in a traditional \st framework. We name our approach ``Fuse, Learn and Distil'' (\afal) framework. \afal aligns student features with expert features through a contrastive knowledge distillation. In order to get the best knowledge from the experts, we introduce a novel Expert Domain Fusion (\edf) module which learns a unified embedding and prevents mode collapse in the presence of source biases. Experiments conducted on four natural imaging datasets and two challenging medical datasets have substantiated the superior performance of \afal compared to existing state-of-the-art methodologies.

# Model Zoo
Refer to [this](https://drive.google.com/drive/folders/1pudcZcTgL05-0ruIxIYHQIVWqKfgMgD6?usp=drive_link) Google Drive link to get access to, Source Pretrained weights of detector and foundation models, Config Files for detector source training and adaptation.

<!--
# Architecture
![architecture](figs/framework.png "model arch")
-->

## Directory structure briefing:

```FUSED/``` Holds the code for Adaptation of FND detector.  
```foundation_code/``` Holds the code for source training and embedding extraction to be used during adaptation.  

Create a ```foundation_weights/``` directory here.  
```foundation_weights/``` will hold the weights and embeddings of foundation models.  

```
foundation_weights/
|
-- vitb16_clip/
|    |-- source_dataset_1/
|           |-- `model_best.pt`
|           |-- target_dataset_1/
|               |-- `embeddings_save.npy`
|           |-- target_dataset_2/
|               |-- `embeddings_save.npy`
-- vitb16_imgnet/
|    |-- source_dataset_1/
|           |-- `model_best.pt`
|           |-- target_dataset_1/
|               |-- `embeddings_save.npy`
|           |-- target_dataset_2/
|               |-- `embeddings_save.npy`
-- vitdino/
    |-- source_dataset_1/
           |-- `model_best.pt`
           |-- target_dataset_1/
                |-- `embeddings_save.npy`
           |-- target_dataset_2/
                |-- `embeddings_save.npy`
```

The files ```model_best.pt``` and ```embeddings_save.npy``` are source trained foundation model and target set embeddings of the foundation model respectively.
For source training and gettings embeddings of foundation models, refer ```foundation_code/README.md``` 

But before doing anything, setup your conda environment following the instructions below:

## INSTALLATION OF ENVIRONMENTS:
In ```foundation_code/``` and ```FUSED/``` lies the .yml config files to create conda environment for running the ```foundation``` and ```adaptation``` code respectively.  
#### 1) Install environment for Foundation models  
```
cd ./foundation_code
conda env create -f found.yml
```
#### 2) Install environment for FocalNet Object detector
Compiling the Deformable Attention Code in ```FUSED/```:  
```
cd ./FUSED
conda env create -f sfda-neurips.yml

cd ./models/dino/ops
python setup.py build install

# unit test (should see all checking is True)
python test.py
```
After these Installations you are good to go...

### Overall Flow of the Code:   
This is the only correct order to run this code:  

### Source Pretraining:
_**1) FND Source pretraining:**_
   Follow through ```./FUSED/README.md``` for FND source training.  
_**2) Foundation Source Pretraining:**_
   Follow through ```./foundation_code/README.md``` for Source training all the foundation models necessary. These weights (`model_best.pt`) will be stored in ```./foundation_weights```.  
   
### Adaptation:
_**1) Get embeddings from foundation models:**_
   Follow through ```./foundation_code/README.md``` for getting the expert/foundation embeddings. These embeddings (`embeddings_save.npy`) should be stored in `./foundation_weights`.  
   After this and source pretraining, the structure of `./foundation_weights` should look like as described above.  
_**2) FUSED Adaptation:**_
   Now, the expert embeddings are used in the adaptation process. Follow through `./FUSED/README.md` for adaptation of the detector.

<!--
   # SFDA-NeurIPS-24

# Abstract
We focus on source-free domain adaptive object detection (\sfdaod), when source data is unavailable during adaptation and the model must adapt on the unlabeled target domain. Majority of approaches for the problem employ a self-supervised approach using a student-teacher (\st) framework. We observe that the performance of a student model often degrades dramatically after \sfdaod, even worse than no domain adaptation. This happens due to teacher model collapsing because of source data bias and large domain shift between source and target domains. On the other hand, vision foundation models (\vfms) such as \clip, \dinov, \sam have demonstrated excellent zero shot performance for various downstream tasks. While trained with distinct objectives and exhibiting unique traits, these models can be unified effectively through multi-expert distillation in a traditional \st framework. We name our approach ``Fuse, Learn and Distil'' (\afal) framework. \afal aligns student features with expert features through a contrastive knowledge distillation. In order to get the best knowledge from the experts, we introduce a novel Expert Domain Fusion (\edf) module which learns a unified embedding and prevents mode collapse in the presence of source biases. Experiments conducted on four natural imaging datasets and two challenging medical datasets have substantiated the superior performance of \afal compared to existing state-of-the-art methodologies.

# Architecture
![architecture](figs/framework.png "model arch")

## Directory structure briefing:

```FUSED/``` Holds the code for Adaptation of FND detector.  
```foundation_code/``` Holds the code for source training and embedding extraction to be used during adaptation.  

Create a ```foundation_weights/``` directory here.  
```foundation_weights/``` will hold the weights and embeddings of foundation models.  

```
foundation_weights/
|
-- vitb16_clip/
|    |-- source_dataset_1/
|           |-- `model_best.pt`
|           |-- target_dataset_1/
|               |-- `embeddings_save.npy`
|           |-- target_dataset_2/
|               |-- `embeddings_save.npy`
-- vitb16_imgnet/
|    |-- source_dataset_1/
|           |-- `model_best.pt`
|           |-- target_dataset_1/
|               |-- `embeddings_save.npy`
|           |-- target_dataset_2/
|               |-- `embeddings_save.npy`
-- vitdino/
    |-- source_dataset_1/
           |-- `model_best.pt`
           |-- target_dataset_1/
                |-- `embeddings_save.npy`
           |-- target_dataset_2/
                |-- `embeddings_save.npy`
```

The files ```model_best.pt``` and ```embeddings_save.npy``` are source trained foundation model and target set embeddings of the foundation model respectively.
For source training and gettings embeddings of foundation models, refer ```foundation_code/README.md``` 

But before doing anything, setup your conda environment following the instructions below:

## INSTALLATION OF ENVIRONMENTS:
In ```foundation_code/``` and ```FUSED/``` lies the .yml config files to create conda environment for running the ```foundation``` and ```adaptation``` code respectively.  

-->

If your `source dataset` is the same as one of our experiments, you may refer to [this](https://drive.google.com/drive/folders/1pudcZcTgL05-0ruIxIYHQIVWqKfgMgD6?usp=drive_link) link to download the concerned source trained foundation models' weights. (`model_best.pt`)

After source training, get the embeddings (by following instructions in ```./foundation_code/README.md```) and use these embeddings in adaptation.
