# A Novel Plug-in Module for Fine-grained Visual Classification

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/a-novel-plug-in-module-for-fine-grained-1/fine-grained-image-classification-on-cub-200)](https://paperswithcode.com/sota/fine-grained-image-classification-on-cub-200?p=a-novel-plug-in-module-for-fine-grained-1)

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/a-novel-plug-in-module-for-fine-grained-1/fine-grained-image-classification-on-nabirds)](https://paperswithcode.com/sota/fine-grained-image-classification-on-nabirds?p=a-novel-plug-in-module-for-fine-grained-1)

paper url: https://arxiv.org/abs/2202.03822 

We propose a novel plug-in module that can be integrated to many common
backbones, including CNN-based or Transformer-based networks to provide strongly discriminative regions. The plugin module can output pixel-level feature maps and fuse filtered features to enhance fine-grained visual classification. Experimental results show that the proposed plugin module outperforms state-ofthe-art approaches and significantly improves the accuracy to **92.77%** and **92.83%** on CUB200-2011 and NABirds, respectively.

![framework](./imgs/0001.png)

### 1. Environment setting 
* install requirements
* replace folder timm/ to our timm/ folder (for ViT or Swin-T)


#### Prepare dataset
In this paper, we use 2 large bird's datasets:
* [CUB-200-2011](http://www.vision.caltech.edu/visipedia/CUB-200-2011.html)
* [NA-Birds](https://dl.allaboutbirds.org/nabirds)

#### Our pretrained model

Download the pretrained model from this url: https://drive.google.com/drive/folders/1ivMJl4_EgE-EVU_5T8giQTwcNQ6RPtAo?usp=sharing      

* backup/ is our pretrained model path.
* resnet50_miil_21k.pth and vit_base_patch16_224_miil_21k.pth are imagenet21k pretrained model (place these file under models/), thanks to https://github.com/Alibaba-MIIL/ImageNet21K/blob/main/MODEL_ZOO.md !!


#### OS
- [x] Windows10
- [x] Ubuntu20.04
- [ ] macOS

### 2. Train
configuration file:  config.py  
```
python train.py --train_root "./CUB200-2011/train/" --val_root "./CUB200-2011/test/"
```

### 3. Evaluation
configuration file:  config_eval.py  
```
python eval.py --pretrained_path "./backup/CUB200/best.pth" --val_root "./CUB200-2011/test/"
```

### 4. Visualization
configuration file:  config_plot.py  
```
python plot_heat.py --pretrained_path "./backup/CUB200/best.pth" --img_path "./img/001.png/"
```
![visualization](./imgs/test1_heat.jpg)


### Acknowledgment

* Thanks to [timm](https://github.com/rwightman/pytorch-image-models) for Pytorch implementation.

* This work was financially supported by the National Taiwan Normal University (NTNU) within the framework of the Higher Education Sprout Project by the Ministry of Education(MOE) in Taiwan, sponsored by Ministry of Science and Technology, Taiwan, R.O.C. under Grant no. MOST 110-
2221-E-003-026, 110-2634-F-003 -007, and 110-2634-F-003 -006. In addition, we thank to National Center for Highperformance Computing (NCHC) for providing computational and storage resources.
