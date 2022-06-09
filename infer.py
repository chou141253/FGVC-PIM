"""
infer version1.0
2022.06.07
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import contextlib
import wandb
import warnings

from models.builder import MODEL_GETTER
from data.dataset import build_loader
from utils.costom_logger import timeLogger
from utils.config_utils import load_yaml, build_record_folder, get_args

warnings.simplefilter("ignore")


def set_environment(args, tlogger):
    print("Setting Environment...")

    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    ### = = = =  Dataset and Data Loader = = = =
    tlogger.print("Building Dataloader....")

    _, val_loader = build_loader(args)

    print("[Only Evaluation]")

    tlogger.print()

    ### = = = =  Model = = = =
    tlogger.print("Building Model....")
    model = MODEL_GETTER[args.model_name](
        use_fpn=args.use_fpn,
        fpn_size=args.fpn_size,
        use_selection=args.use_selection,
        num_classes=args.num_classes,
        num_selects=args.num_selects,
        use_combiner=args.use_combiner,
    )  # about return_nodes, we use our default setting
    print(model)

    checkpoint = torch.load(args.pretrained, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['model'])
    start_epoch = checkpoint['epoch']


    # model = torch.nn.DataParallel(model, device_ids=None) # device_ids : None --> use all gpus.
    model.to(args.device)
    tlogger.print()

    """
    if you have multi-gpu device, you can use torch.nn.DataParallel in single-machine multi-GPU 
    situation and use torch.nn.parallel.DistributedDataParallel to use multi-process parallelism.
    more detail: https://pytorch.org/tutorials/beginner/dist_overview.html
    """


    return val_loader, model


def main_test(args, tlogger):
    """
    infer and confusion matrix
    """

    val_loader, model = set_environment(args, tlogger)
    from eval import eval_and_cm
    eval_and_cm(args, model, val_loader, tlogger)


if __name__ == "__main__":
    tlogger = timeLogger()

    tlogger.print("Reading Config...")
    args = get_args()
    assert args.c != "", "Please provide config file (.yaml)"
    load_yaml(args, args.c)
    build_record_folder(args)
    tlogger.print()

    main_test(args, tlogger)