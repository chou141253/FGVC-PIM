import argparse
import os

def get_args():

    parser = argparse.ArgumentParser("Plot heatmap")
    # save path and dataset information
    parser.add_argument("--threshold", default=0.75, type=float, 
        help="heatmap threshold")

    parser.add_argument("--pretrained_path", default="./backup/CUB200/best.pth", type=str)
    parser.add_argument("--img_path", default="./plot_imgs/test1.jpg", type=str)
    parser.add_argument("--target_class", default=52, type=int)
    parser.add_argument("--data_size", default=384, type=int)
    parser.add_argument("--model_name", default="swin-vit-p4w12", type=str, 
        choices=["efficientnet-b7", 'resnet-50', 'vit-b16', 'swin-vit-p4w12'])
    
    # = = = = = building model = = = = = 
    parser.add_argument("--use_fpn", default=True, type=bool)
    parser.add_argument("--use_ori", default=False, type=bool)
    parser.add_argument("--use_gcn", default=True, type=bool)
    parser.add_argument("--use_layers", 
        default=[True, True, True, True], type=list)
    parser.add_argument("--use_selections", 
        default=[True, True, True, True], type=list)
    parser.add_argument("--num_selects",
        default=[2048, 512, 128, 32], type=list)
    parser.add_argument("--global_feature_dim", default=1536, type=int)
    parser.add_argument("--num_classes", default=200, type=int)

    args = parser.parse_args()

    return args
