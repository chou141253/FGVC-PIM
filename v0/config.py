import argparse
import os

def get_args():

    parser = argparse.ArgumentParser("FineGrained Image Classification Task")
    # save path and dataset information
    parser.add_argument("--exp_name", default="CUB200#SwinVit@TWCC1-GCN1-005")
    
    parser.add_argument("--train_root", 
        default="../datas/train/", type=str) # "../NABirds/train/"
    parser.add_argument("--val_root", 
        default="../datas/test/", type=str)
    parser.add_argument("--data_size", default=384, type=int)
    parser.add_argument("--num_rows", default=0, type=int)
    parser.add_argument("--num_cols", default=0, type=int)
    parser.add_argument("--sub_data_size", default=32, type=int)

    parser.add_argument("--model_name", default="swin-vit-p4w12", type=str, 
        choices=["efficientnet-b7", 'resnet-50', 'vit-b16', 'swin-vit-p4w12'])
    parser.add_argument("--optimizer_name", default="sgd", type=str, 
        choices=["sgd", 'adamw'])
    
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
    
    # loader
    parser.add_argument("--num_workers", default=2, type=int)
    parser.add_argument("--batch_size", default=8, type=int)
    
    # about model building
    parser.add_argument("--num_classes", default=200, type=int)
    
    # abput learning rate scheduler
    parser.add_argument("--warmup_batchs", default=800, type=int)
    parser.add_argument("--no_final_epochs", default=0, type=int)
    parser.add_argument("--max_lr", default=0.0005, type=float)
    parser.add_argument("--update_freq", default=4, type=int)
    
    parser.add_argument("--wdecay", default=0.0005, type=float)
    parser.add_argument("--nesterov", default=True, type=bool)
    parser.add_argument("--max_epochs", default=50, type=int)

    parser.add_argument("--log_freq", default=20, type=int)

    parser.add_argument("--test_freq", default=5, type=int)
    parser.add_argument("--test_global_top_confs", default=[1, 3, 5], type=list)
    parser.add_argument("--test_select_top_confs", default=[1, 3, 5, 7, 9], type=list)

    args = parser.parse_args()
    args = build_record_folder(args)

    return args


def build_record_folder(args):
    print("building records folder...", end="")
    
    if not os.path.isdir("./records/"):
        print(".../records/...", end="")
        os.mkdir("./records/")
    args.save_root = "./records/" + args.exp_name + "/"
    os.makedirs(args.save_root, exist_ok=True)
    
    print("...{}...".format(args.save_root), end="")
    
    # save labeled images path and unlabeled images path.
    os.makedirs(args.save_root + "data_info/", exist_ok=True)
    os.makedirs(args.save_root + "backup/", exist_ok=True)
    os.makedirs(args.save_root + "distributions/", exist_ok=True)
    
    print("...{}...".format(args.save_root + "x_ux_info/"), end="")
    print("...finish")
    print()
    return args
