# Foundation Models Guide Best: Source-free Domain Adaptive Object Detection via Contrastive Knowledge Distillation

### FIRST THINGS FIRST
- Follow through the environment installation in the repo's root, and activate ```sfda-neurips``` environment.

# SETTING UP:  
1) Download FND backbone from [this](https://drive.google.com/drive/folders/1cn_JXrFa2uCNNxC-692_74GzTGPtXMtY?usp=drive_link) link.
2) Download `weights_template/` folder here in this `FUSED` directory. Then change it's name to `weights/`. This weights folder contains the config files required for source training and adaptation. For the settings used in our paper, the config files here were used exactly for our experiments.
_**Note:**_ You may source train the FND detector and put it appropriately in the `weights/` folder or you may download the concerned source trained FND weights from [this](https://drive.google.com/drive/folders/1cn_JXrFa2uCNNxC-692_74GzTGPtXMtY?usp=drive_link) link, from `./weights_completed`.

# Running the code
## (I) SOURCE PRETRAINING
Source training of FND. (Refer readme in ```foundation_code/``` for Foundation source pretraining)
  
```sh
CUDA_VISIBLE_DEVICES=1 python -m torch.distributed.launch --nproc_per_node=1 --master_port 11002 main.py \
--config_file ./weights/source_aiims/config_cfg.py \
--source_dataset inhouse --target_dataset inbreast \
--coco_path <source_dataset_path>
--output_dir ./weights/source_aiims \
--dataset_file coco > srctrainlogs.txt
```
The source trained checkpoint is selected by the best performance on source data. Full training is done till 100 epochs.

**Now refer to `../foundation_code/README.md` for next steps.**

## (II) TARGET ADAPTATION  
_**NOTE:**_
- Make sure you have `sfda-neurips` environment activated.
- Make sure that you completely get the embeddings for your foundation models in ```../foundation_weights``` folder. Currently the code supports the incorporation of 3 foundation models, but it can be extended depending upon the user's use cases and resources.
- Make sure you have FocalNet backbone in this ```./FUSED``` directory. You can download from [this](https://drive.google.com/drive/folders/1cn_JXrFa2uCNNxC-692_74GzTGPtXMtY?usp=drive_link) link.

```
CUDA_VISIBLE_DEVICES=1 python -m torch.distributed.launch --nproc_per_node=1 --master_port 11002 main.py \
--config_file ./weights/source_inhouse/target_inbreast/config_cfg.py --source_dataset inhouse \
--target_dataset inbreast \
--coco_path /home/keerti_km/scratch/Negroni_datasets/Coco_Data/BCD \
--pretrain_model_path ./weights/source_inhouse/source_in.pth --output_dir ./weights/source_inhouse/ \
--find_unused_params --module_type student_with_expert_module --adapt --temp 1.0 --memory_bank_size 1 \
--lambda_eckd 1.0 --init_threshold 0.06 \
--expert_embed_dir ../foundation_weights \
--conf_update_algo const_thresh
```

## (III) EVALUATION  
_**NOTE:**_
- Make sure you have `sfda-neurips` environment activated.
### a) Natural setting evaluation
```sh
  bash test.sh /path/to/output/dir /path/to/config /path/to/dataset /path/to/checkpoint 1.0 1.0
```
The last two values are dummy variables - as in, they don't effect the evaluation of model but are deemed required by the code so that
they are not forgotten while during adaptation - for which they play a great role in.

### b) Medical setting evaluation
```sh
  python medical_evaluation.py
```






<!-- 
# Abstract
We focus on source-free domain adaptive object detection (\sfdaod), when source data is unavailable during adaptation and the model must adapt on the unlabeled target domain. Majority of approaches for the problem employ a self-supervised approach using a student-teacher (\st) framework. We observe that the performance of a student model often degrades dramatically after \sfdaod, even worse than no domain adaptation. This happens due to teacher model collapsing because of source data bias and large domain shift between source and target domains. On the other hand, vision foundation models (\vfms) such as \clip, \dinov, \sam have demonstrated excellent zero shot performance for various downstream tasks. While trained with distinct objectives and exhibiting unique traits, these models can be unified effectively through multi-expert distillation in a traditional \st framework. We name our approach ``Fuse, Learn and Distil'' (\afal) framework. \afal aligns student features with expert features through a contrastive knowledge distillation. In order to get the best knowledge from the experts, we introduce a novel Expert Domain Fusion (\edf) module which learns a unified embedding and prevents mode collapse in the presence of source biases. Experiments conducted on four natural imaging datasets and two challenging medical datasets have substantiated the superior performance of \afal compared to existing state-of-the-art methodologies.














![teaser](figs/teaser.jpg "Problem with ST framework and SOTA results")

# Architecture
![architecture](figs/framework.png "model arch")

# Model Zoo
We provide FocalNet Large FL4 backbone, Source trained and Adapted models along with the corresponding config files and Expert embeddings on [[Google Drive]](https://drive.google.com/drive/folders/1qD5m1NmK0kjE5hh-G17XUX751WsEG-h_?usp=sharing). Place the backbone weights in the root directory before adaptation or source training.
-->
<!-- 
# Installation

<details>
  <summary>Installation</summary>
  
   We test our models under ```python=3.9.19,pytorch=1.11.0,cuda=11.3.1```.

   1. Clone this repo
   ```sh
   git clone https://github.com/IDEA-Research/DINO.git
   cd DINO
   ```

   2. Install Pytorch and torchvision

   Follow the instruction on https://pytorch.org/get-started/previous-versions/.
   ```sh
   # an example:
   conda install pytorch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0 cudatoolkit=11.3 -c pytorch
   ```

   3. Install other needed packages
   ```sh
   pip install -r requirements.txt
   ```

   4. Compiling CUDA operators
   ```sh
   cd models/dino/ops
   python setup.py build install
   # unit test (should see all checking is True)
   python test.py
   cd ../../..
   ```
</details>
# Data

<details>
  <summary>Data</summary>

You can download the datasets from official websites: [cityscapes](https://www.cityscapes-dataset.com/downloads/),  [foggy_cityscapes](https://www.cityscapes-dataset.com/downloads/),  [sim10k](https://fcav.engin.umich.edu/projects/driving-in-the-matrix), [bdd100k](https://bdd-data.berkeley.edu/). You can also download coco converted annotations from [here](https://drive.google.com/file/d/1LB0wK9kO3eW8jpR2ZtponmYWe9x2KSiU/view?usp=sharing) and organize the datasets and annotations as following:
```
datasets
|
COCODIR/
  ├── train2017/
  ├── val2017/
  └── annotations/
  	├── instances_train2017.json
  	└── instances_val2017.json
```

</details>


# Run

<details>
  <summary>Some Notes on Adaptation</summary>
  <!-- ### Eval our pretrianed model -->
 <!-- You might want to tune confidence updating algorithm (`args.conf_update_algo`) depending on the number of pseudo labels for a target dataset.  
  We provide three algorithms to choose from:
  `const_thresh`: Pseudo label Threshold is kept constant throughout the epochs.
  `raise_slowly`: This algorithm raises the threshold slowly with by some small margin, say by 0.1 / num_epochs
  `raise_abruptly`: This algorithm raises the threshold in next epoch abruptly to a value: (avg_conf + highest_conf) / 2, where avg_conf and highest_conf are average
                    and highest confidences of the pseudo labels that appeared in the last epoch.
  We found that it's usually convenient to use raise_slowly algorithm in natural setting and raise_abruptly in breast cancer detection setting.
</details>

<details>
  <summary>1. Eval our pretrianed models</summary>

  <!-- ### Eval our pretrianed model -->

  <!--```sh
  bash test.sh /path/to/output/dir /path/to/config /path/to/dataset /path/to/checkpoint
  ```-->

<!--</details>



<details>
  <summary>2. Inference and Visualizations</summary>

For inference and visualizations, we provide a [notebook](inference_and_visualization.ipynb) as an example.

</details>



<details>
  <summary>3. Adapt DINO 4-scale mode with FocalNet Large - 4 backbone  </summary>
  
```sh
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --master_port 11002 main.py \
--config_file ./weights/path/to/config/file --target_dataset <target_dataset> --source_dataset <source_dataset> \
--coco_path /path/to/dataset \
--pretrain_model_path /path/to/source/trained/checkpoint \
--output_dir ./exps/ \
--module_type student_with_expert_module \
--adapt --init_threshold 0.06 --topk_pseudo 20 \
--expert_embed_dir ./expert_embeddings/
```

</details>


# Results
## Natural

## Breast Cancer Detection
-->

## ACKNOWLEDGEMENTS
We also thank great previous work including DETR, Deformable DETR, SMCA, Conditional DETR, Anchor DETR, Dynamic DETR, etc. More related work are available at [Awesome Detection Transformer](https://github.com/IDEACVR/awesome-detection-transformer).

## LICNESE
DINO is released under the Apache 2.0 license. Please see the [LICENSE](LICNESE) file for more information.

Copyright (c) IDEA. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use these files except in compliance with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.
