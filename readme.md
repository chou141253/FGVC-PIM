
# A Novel Plug-in Module for Fine-grained Visual Classification

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/a-novel-plug-in-module-for-fine-grained-1/fine-grained-image-classification-on-cub-200)](https://paperswithcode.com/sota/fine-grained-image-classification-on-cub-200?p=a-novel-plug-in-module-for-fine-grained-1)

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/a-novel-plug-in-module-for-fine-grained-1/fine-grained-image-classification-on-nabirds)](https://paperswithcode.com/sota/fine-grained-image-classification-on-nabirds?p=a-novel-plug-in-module-for-fine-grained-1)

paper url: https://arxiv.org/abs/2202.03822 

We propose a novel plug-in module that can be integrated to many common
backbones, including CNN-based or Transformer-based networks to provide strongly discriminative regions. The plugin module can output pixel-level feature maps and fuse filtered features to enhance fine-grained visual classification. Experimental results show that the proposed plugin module outperforms state-ofthe-art approaches and significantly improves the accuracy to **92.77%** and **92.83%** on CUB200-2011 and NABirds, respectively.

![framework](./imgs/0001.png)

## 1. Environment setting 

// We move old version to ./v0/

### 1.0. Package
* install requirements
* replace folder timm/ to our timm/ folder (for ViT or Swin-T)  
    
    #### pytorch model implementation [timm](https://github.com/rwightman/pytorch-image-models)
    #### recommand [anaconda](https://www.anaconda.com/products/distribution)
    #### recommand [weights and biases](https://wandb.ai/site)
    #### [deepspeed](https://www.deepspeed.ai/getting-started/) // future works

### 1.1. Dataset
In this paper, we use 2 large bird's datasets to evaluate performance:
* [CUB-200-2011](http://www.vision.caltech.edu/visipedia/CUB-200-2011.html)
* [NA-Birds](https://dl.allaboutbirds.org/nabirds)

### 1.2. Our pretrained model

* our pretrained model in https://idocntnu-my.sharepoint.com/:f:/g/personal/81075001h_eduad_ntnu_edu_tw/EkypiS-W0SFDkxnHN1Imv5oBPgoRblDgW8wHuVA0c6Ka7Q?e=FhBLDC
* cub200 and nabird dataset: https://idocntnu-my.sharepoint.com/:f:/g/personal/81075001h_eduad_ntnu_edu_tw/EoBb2gijwclEulDGxv_hOtIBeKuV3M6qy3IGIGMhm-jq0g?e=tcg6tm
* resnet50_miil_21k.pth and vit_base_patch16_224_miil_21k.pth are imagenet21k pretrained model (place these file under models/), thanks to https://github.com/Alibaba-MIIL/ImageNet21K/blob/main/MODEL_ZOO.md !!

### 1.3. OS
- [x] Windows10
- [x] Ubuntu20.04
- [x] macOS (CPU only)

## 2. Train
- [x] Single GPU Training
- [x] DataParallel (single machine multi-gpus)
- [ ] DistributedDataParallel

(more information: https://pytorch.org/tutorials/intermediate/ddp_tutorial.html)

### 2.1. data
train data and test data structure:  
```
├── tain/
│   ├── class1/
│   |   ├── img001.jpg
│   |   ├── img002.jpg
│   |   └── ....
│   ├── class2/
│   |   ├── img001.jpg
│   |   ├── img002.jpg
│   |   └── ....
│   └── ....
└──
```

### 2.2. configuration
you can directly modify yaml file (in ./configs/)

### 2.3. run
```
python main.py --c ./configs/CUB200_SwinT.yaml
```
model will save in ./records/{project_name}/{exp_name}/backup/


### 2.4. about costom model
Building model refers to ./models/builder.py   
More detail in [how_to_build_pim_model.ipynb](./how_to_build_pim_model.ipynb)

### 2.5. multi-gpus
comment out main.py line 66
```
model = torch.nn.DataParallel(model, device_ids=None)
```

### 2.6.  automatic mixed precision (amp)
use_amp: True, training time about 3-hours.  
use_amp: False, training time about 5-hours.  

## 3. Evaluation
If you want to evaluate our pretrained model (or your model), please give provide configs/eval.yaml (or costom yaml file is fine)

### 3.1. please check yaml
set yaml (configuration file)
Key           | Value  | Description | 
--------------|:------|:------------| 
train_root    | ~      | set value to ~ (null) means this is not in training mode.  |
val_root  | ../data/eval/  |  path to validation samples |
pretrained  | ./pretrained/best.pt  |   pretrained model path |


../data/eval/ folder structure:  
```
├── eval/
│   ├── class1/
│   |   ├── img001.jpg
│   |   ├── img002.jpg
│   |   └── ....
│   ├── class2/
│   |   ├── img001.jpg
│   |   ├── img002.jpg
│   |   └── ....
│   └── ....
└──
```

### 3.2. run
```
python main.py --c ./configs/eval.yaml
```
results will show in terminal and been save in ./records/{project_name}/{exp_name}/eval_results.txt

## 4. HeatMap
```
python heat.py --c ./configs/CUB200_SwinT.yaml --img ./vis/001.jpg --save_img ./vis/001/
```
![visualization](./vis/001/rbg_img.jpg)
![visualization2](./vis/001/mix.jpg)

## 5. Infer
If you want to reason your picture and get the confusion matrix, please give provide configs/eval.yaml (or costom yaml file is fine)


### 5.1. please check yaml
set yaml (configuration file)
Key           | Value  | Description | 
--------------|:------|:------------| 
train_root    | ~      | set value to ~ (null) means this is not in training mode.  |
val_root  | ../data/eval/  |  path to validation samples |
pretrained  | ./pretrained/best.pt  |   pretrained model path |


../data/eval/ folder structure:  
```
├── eval/
│   ├── class1/
│   |   ├── img001.jpg
│   |   ├── img002.jpg
│   |   └── ....
│   ├── class2/
│   |   ├── img001.jpg
│   |   ├── img002.jpg
│   |   └── ....
│   └── ....
└──
```

### 5.2. run
```
python infer.py --c ./configs/eval.yaml
```
results will show in terminal and been save in ./records/{project_name}/{exp_name}/infer_results.txt

- - - - - - 

### Acknowledgment

* Thanks to [timm](https://github.com/rwightman/pytorch-image-models) for Pytorch implementation.

* This work was financially supported by the National Taiwan Normal University (NTNU) within the framework of the Higher Education Sprout Project by the Ministry of Education(MOE) in Taiwan, sponsored by Ministry of Science and Technology, Taiwan, R.O.C. under Grant no. MOST 110-
2221-E-003-026, 110-2634-F-003 -007, and 110-2634-F-003 -006. In addition, we thank to National Center for Highperformance Computing (NCHC) for providing computational and storage resources.