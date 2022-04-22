import yaml
import os
import argparse

def load_yaml(args, yml):
    with open(yml, 'r', encoding='utf-8') as fyml:
        dic = yaml.load(fyml.read(), Loader=yaml.Loader)
        for k in dic:
            setattr(args, k, dic[k])

def build_record_folder(args):

    if not os.path.isdir("./records/"):
        os.mkdir("./records/")
    
    args.save_dir = "./records/" + args.project_name + "/" + args.exp_name + "/"
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.save_dir + "backup/", exist_ok=True)

def get_args(with_deepspeed: bool=False):

    parser = argparse.ArgumentParser("Fine-Grained Visual Classification")

    parser.add_argument("--project_name", default="")
    parser.add_argument("--exp_name", default="")

    parser.add_argument("--c", default="", type=str, help="config file path")
    
    ### about dataset
    parser.add_argument("--train_root", default="", type=str) # "../NABirds/train/"
    parser.add_argument("--val_root", default="", type=str)
    parser.add_argument("--data_size", default=384, type=int)
    parser.add_argument("--num_workers", default=2, type=int)
    parser.add_argument("--batch_size", default=64, type=int)

    ### model
    parser.add_argument("--model_name", default="", type=str, help='["resnet50", "swin-t", "vit", "efficient"]')
    parser.add_argument("--optimizer", default="", type=str, help='["SGD", "AdamW"]')
    parser.add_argument("--max_lr", default=0.0003, type=float)
    parser.add_argument("--wdecay", default=0.0005, type=float)
    
    parser.add_argument("--max_epochs", default=50, type=int)
    parser.add_argument("--warmup_batchs", default=0, type=int)

    parser.add_argument("--use_fpn", default=True, type=bool)
    parser.add_argument("--fpn_size", default=512, type=int)
    parser.add_argument("--use_selection", default=True, type=bool)
    parser.add_argument("--num_classes", default=10, type=int)
    parser.add_argument("--num_selects", default={
            "layer1":32,
            "layer2":32,
            "layer3":32,
            "layer4":32
        }, type=dict)
    parser.add_argument("--use_combiner", default=True, type=bool)

    ### loss
    parser.add_argument("--lambda_b", default=0.5, type=float)
    parser.add_argument("--lambda_s", default=0.0, type=float)
    parser.add_argument("--lambda_n", default=5.0, type=float)
    parser.add_argument("--lambda_c", default=1.0, type=float)

    parser.add_argument("--use_wandb", default=True, type=bool)

    if with_deepspeed:
        import deepspeed
        parser = deepspeed.add_config_arguments(parser)
    
    args = parser.parse_args()

    return args

