import os
import wandb
import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import numpy as np
import math
import copy
import cv2
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt

from config_plot import get_args


def set_environment(args):

    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    if args.model_name == 'swin-vit-p4w12':
        from models.SwinVit12_demo import SwinVit12
        model = SwinVit12(
                in_size=args.data_size,
                num_classes=args.num_classes, 
                use_fpn=args.use_fpn, 
                use_ori=args.use_ori,
                use_gcn=args.use_gcn,
                use_layers=args.use_layers,
                use_selections=args.use_selections,
                num_selects=args.num_selects,
                global_feature_dim=args.global_feature_dim
            )

    checkpoint = torch.load(args.pretrained_path)
    model.load_state_dict(checkpoint['model'])
    model.to(args.device)

    # convert image to tensor
    img_transforms = transforms.Compose([
            transforms.Resize((510, 510), Image.BILINEAR),
            transforms.CenterCrop((args.data_size, args.data_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
    ])
    img = cv2.imread(args.img_path)
    img = img[:, :, ::-1] # BGR to RGB.
    
    # to PIL.Image
    img = Image.fromarray(img)
    img = img_transforms(img)
    
    return img, model


def simple_grad_cam(features, classifier, target_class):
    """
    calculate gradient map.
    """
    features = nn.Parameter(features)

    logits = torch.matmul(features, classifier)
    
    logits[0, :, :, target_class].sum().backward()
    features_grad = features.grad[0].sum(0).sum(0).unsqueeze(0).unsqueeze(0)
    gramcam = F.relu(features_grad * features[0])
    gramcam = gramcam.sum(-1)
    gramcam = (gramcam - torch.min(gramcam)) / (torch.max(gramcam) - torch.min(gramcam))

    return gramcam

def generate_heat(args, features:list):
    w = [8, 4, 2, 1]
    heatmap = np.zeros([args.data_size, args.data_size, 3])
    for fi in range(len(features)):
        f = features[fi]
        f = f.cpu()[0]
        S = int(f.size(0) ** 0.5)
        f = f.view(S, S, -1)
        # if use original backbone without our module, 
        # please set classifier to your model's classifier.
        gramcam = simple_grad_cam(f.unsqueeze(0), 
                                  classifier=torch.ones(f.size(-1), 200)/f.size(-1), 
                                  target_class=args.target_class)
        gramcam = gramcam.detach().numpy()
        gramcam = cv2.resize(gramcam, (args.data_size, args.data_size))
        # heat: red
        heatmap[:, :, 2] += w[fi]*gramcam

    heatmap = heatmap / sum(w)
    heatmap = (heatmap-heatmap.min()) / (heatmap.max() - heatmap.min())
    heatmap[heatmap<args.threshold] = 0 # threshold
    heatmap *= 255
    heatmap = heatmap.astype(np.uint8)

    return heatmap
    

def main(args, img, model):
    
    model.eval()

    """ read original image """
    rgb_img = cv2.imread(args.img_path)
    rgb_img = cv2.resize(rgb_img, (510, 510))
    pad_size = (510 - 384) // 2
    rgb_img = rgb_img[pad_size:-pad_size, pad_size:-pad_size]

    """ forward """
    img = img.to(args.device)
    with torch.no_grad():
        features = model(img.unsqueeze(0))

    heatmap = generate_heat(args, features)

    plt_img = rgb_img * 0.5 + heatmap * 0.5
    plt_img = plt_img.astype(np.uint8)

    cv2.namedWindow('heatmap', 0)
    cv2.imshow('heatmap', plt_img)
    save_path = args.img_path.replace('.'+args.img_path.split('.')[-1], "") + \
        "_heat.jpg"
    cv2.imwrite(save_path, plt_img)
    cv2.waitKey(0)



if __name__ == "__main__":
    args = get_args()
    test_img, model = set_environment(args)
    main(args, test_img, model)
