import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import argparse
import warnings

from utils.config_utils import load_yaml
from models.builder import MODEL_GETTER
from utils.costom_logger import timeLogger

warnings.simplefilter("ignore")

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


def get_heat(model, img):
    # only need forward backbone

    with torch.no_grad():
        outs = model.forward_backbone(img.unsqueeze(0))
    
    features = []
    for name in outs:
        features.append(outs[name][0])

    layer_weights = [8, 4, 2, 1]
    heatmap = np.zeros([args.data_size, args.data_size, 3])
    for i in range(len(features)):
        f = features[i]
        f = f.cpu()
        if len(f.size()) == 2:
            S = int(f.size(0) ** 0.5)
            f = f.view(S, S, -1)

        # if you use original backbone without our module, 
        # please set classifier to your model's classifier. (e.g. model.classifier)
        gramcam = simple_grad_cam(f.unsqueeze(0), classifier=torch.ones(f.size(-1), 200)/f.size(-1), target_class=args.target_class)
        gramcam = gramcam.detach().numpy()
        gramcam = cv2.resize(gramcam, (args.data_size, args.data_size))
        # heatmap colour : red
        heatmap[:, :, 2] += layer_weights[i] * gramcam

    heatmap = heatmap / sum(layer_weights)
    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())
    heatmap[heatmap < args.threshold] = 0 # threshold
    heatmap *= 255
    heatmap = heatmap.astype(np.uint8)

    return heatmap

if __name__ == "__main__":

    parser = argparse.ArgumentParser("PIM-FGVC Heatmap Generation")
    parser.add_argument("--c", default="", type=str)
    parser.add_argument("--img", default="", type=str)
    parser.add_argument("--target_class", default=0, type=int)
    parser.add_argument("--threshold", default=0.75, type=float)
    parser.add_argument("--save_img", default="", type=str, help="save path")
    parser.add_argument("--pretrained", default="", type=str)
    parser.add_argument("--model_name", default="swin-t", type=str, choices=["swin-t", "resnet50", "vit", "efficient"])
    args = parser.parse_args()

    assert args.c != "", "Please provide config file (.yaml)"

    args = parser.parse_args()
    load_yaml(args, args.c)

    assert args.pretrained != ""

    model = MODEL_GETTER[args.model_name](
        use_fpn = args.use_fpn,
        fpn_size = args.fpn_size,
        use_selection = args.use_selection,
        num_classes = args.num_classes,
        num_selects = args.num_selects,
        use_combiner = args.use_combiner,
    ) # about return_nodes, we use our default setting

    ### load model
    checkpoint = torch.load(args.pretrained, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['model'])
    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(args.device)

    ### read image and convert image to tensor
    img_transforms = transforms.Compose([
            transforms.Resize((510, 510), Image.BILINEAR),
            transforms.CenterCrop((args.data_size, args.data_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
    ])
    img = cv2.imread(args.img)
    img = img[:, :, ::-1] # BGR to RGB.
    
    # to PIL.Image
    img = Image.fromarray(img)
    img = img_transforms(img)
    img = img.to(args.device)

    # get heatmap and original image
    heatmap = get_heat(model, img)
    
    rgb_img = cv2.imread(args.img)
    rgb_img = cv2.resize(rgb_img, (510, 510))
    pad_size = (510 - args.data_size) // 2
    rgb_img = rgb_img[pad_size:-pad_size, pad_size:-pad_size]

    mix = rgb_img * 0.5 + heatmap * 0.5
    mix = mix.astype(np.uint8)

    # cv2.namedWindow('heatmap', 0)
    # cv2.imshow('heatmap', heatmap)
    # cv2.namedWindow('rgb_img', 0)
    # cv2.imshow('rgb_img', rgb_img)
    # cv2.namedWindow('mix', 0)
    # cv2.imshow('mix', mix)
    # cv2.watiKey(0)

    if args.save_img != "":
        cv2.imwrite(args.save_img + "/heatmap.jpg", heatmap)
        cv2.imwrite(args.save_img + "/rbg_img.jpg", rgb_img)
        cv2.imwrite(args.save_img + "/mix.jpg", mix)
