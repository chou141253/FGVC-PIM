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
from utils.lr_schedule import cosine_decay, adjust_lr, get_lr
from eval import evaluate, cal_train_metrics

warnings.simplefilter("ignore")

def eval_freq_schedule(args, epoch: int):
    if epoch >= args.max_epochs * 0.95:
        args.eval_freq = 1
    elif epoch >= args.max_epochs * 0.9:
        args.eval_freq = 1
    elif epoch >= args.max_epochs * 0.8:
        args.eval_freq = 2

def set_environment(args, tlogger):
    
    print("Setting Environment...")

    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    ### = = = =  Dataset and Data Loader = = = =  
    tlogger.print("Building Dataloader....")
    
    train_loader, val_loader = build_loader(args)
    
    if train_loader is None and val_loader is None:
        raise ValueError("Find nothing to train or evaluate.")

    if train_loader is not None:
        print("    Train Samples: {} (batch: {})".format(len(train_loader.dataset), len(train_loader)))
    else:
        # raise ValueError("Build train loader fail, please provide legal path.")
        print("    Train Samples: 0 ~~~~~> [Only Evaluation]")
    if val_loader is not None:
        print("    Validation Samples: {} (batch: {})".format(len(val_loader.dataset), len(val_loader)))
    else:
        print("    Validation Samples: 0 ~~~~~> [Only Training]")
    tlogger.print()

    ### = = = =  Model = = = =  
    tlogger.print("Building Model....")
    model = MODEL_GETTER[args.model_name](
        use_fpn = args.use_fpn,
        fpn_size = args.fpn_size,
        use_selection = args.use_selection,
        num_classes = args.num_classes,
        num_selects = args.num_selects,
        use_combiner = args.use_combiner,
    ) # about return_nodes, we use our default setting
    if args.pretrained is not None:
        checkpoint = torch.load(args.pretrained, map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint['model'])
        start_epoch = checkpoint['epoch']
    else:
        start_epoch = 0

    # model = torch.nn.DataParallel(model, device_ids=None) # device_ids : None --> use all gpus.
    model.to(args.device)
    tlogger.print()
    
    """
    if you have multi-gpu device, you can use torch.nn.DataParallel in single-machine multi-GPU 
    situation and use torch.nn.parallel.DistributedDataParallel to use multi-process parallelism.
    more detail: https://pytorch.org/tutorials/beginner/dist_overview.html
    """
    
    if train_loader is None:
        return train_loader, val_loader, model, None, None, None, None
    
    ### = = = =  Optimizer = = = =  
    tlogger.print("Building Optimizer....")
    if args.optimizer == "SGD":
        optimizer = torch.optim.SGD(model.parameters(), lr=args.max_lr, nesterov=True, momentum=0.9, weight_decay=args.wdecay)
    elif args.optimizer == "AdamW":
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.max_lr)

    if args.pretrained is not None:
        optimizer.load_state_dict(checkpoint['optimizer'])

    tlogger.print()

    schedule = cosine_decay(args, len(train_loader))

    if args.use_amp:
        scaler = torch.cuda.amp.GradScaler()
        amp_context = torch.cuda.amp.autocast
    else:
        scaler = None
        amp_context = contextlib.nullcontext

    return train_loader, val_loader, model, optimizer, schedule, scaler, amp_context, start_epoch


def train(args, epoch, model, scaler, amp_context, optimizer, schedule, train_loader):
    
    optimizer.zero_grad()
    total_batchs = len(train_loader) # just for log
    show_progress = [x/10 for x in range(11)] # just for log
    progress_i = 0
    for batch_id, (ids, datas, labels) in enumerate(train_loader):
        model.train()
        """ = = = = adjust learning rate = = = = """
        iterations = epoch * len(train_loader) + batch_id
        adjust_lr(iterations, optimizer, schedule)

        batch_size = labels.size(0)

        """ = = = = forward and calculate loss = = = = """
        datas, labels = datas.to(args.device), labels.to(args.device)

        with amp_context():
            """
            [Model Return]
                FPN + Selector + Combiner --> return 'layer1', 'layer2', 'layer3', 'layer4', ...(depend on your setting)
                    'preds_0', 'preds_1', 'comb_outs'
                FPN + Selector --> return 'layer1', 'layer2', 'layer3', 'layer4', ...(depend on your setting)
                    'preds_0', 'preds_1'
                FPN --> return 'layer1', 'layer2', 'layer3', 'layer4' (depend on your setting)
                ~ --> return 'ori_out'
            
            [Retuen Tensor]
                'preds_0': logit has not been selected by Selector.
                'preds_1': logit has been selected by Selector.
                'comb_outs': The prediction of combiner.
            """
            outs = model(datas)

            loss = 0.
            for name in outs:
                
                if "select_" in name:
                    if not args.use_selection:
                        raise ValueError("Selector not use here.")
                    if args.lambda_s != 0:
                        S = outs[name].size(1)
                        logit = outs[name].view(-1, args.num_classes).contiguous()
                        loss_s = nn.CrossEntropyLoss()(logit, 
                                                       labels.unsqueeze(1).repeat(1, S).flatten(0))
                        loss += args.lambda_s * loss_s
                    else:
                        loss_s = 0.0

                elif "drop_" in name:
                    if not args.use_selection:
                        raise ValueError("Selector not use here.")

                    if args.lambda_n != 0:
                        S = outs[name].size(1)
                        logit = outs[name].view(-1, args.num_classes).contiguous()
                        n_preds = nn.Tanh()(logit)
                        labels_0 = torch.zeros([batch_size * S, args.num_classes]) - 1
                        labels_0 = labels_0.to(args.device)
                        loss_n = nn.MSELoss()(n_preds, labels_0)
                        loss += args.lambda_n * loss_n
                    else:
                        loss_n = 0.0

                elif "layer" in name:
                    if not args.use_fpn:
                        raise ValueError("FPN not use here.")
                    if args.lambda_b != 0:
                        ### here using 'layer1'~'layer4' is default setting, you can change to your own
                        loss_b = nn.CrossEntropyLoss()(outs[name].mean(1), labels)
                        loss += args.lambda_b * loss_b
                    else:
                        loss_b = 0.0
                
                elif "comb_outs" in name:
                    if not args.use_combiner:
                        raise ValueError("Combiner not use here.")

                    if args.lambda_c != 0:
                        loss_c = nn.CrossEntropyLoss()(outs[name], labels)
                        loss += args.lambda_c * loss_c

                elif "ori_out" in name:
                    loss_ori = F.cross_entropy(outs[name], labels)
                    loss += loss_ori
            
            loss /= args.update_freq
        
        """ = = = = calculate gradient = = = = """
        if args.use_amp:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        """ = = = = update model = = = = """
        if (batch_id + 1) % args.update_freq == 0:
            if args.use_amp:
                scaler.step(optimizer)
                scaler.update() # next batch
            else:
                optimizer.step()
            optimizer.zero_grad()

        """ log (MISC) """
        if args.use_wandb and ((batch_id + 1) % args.log_freq == 0):
            model.eval()
            msg = {}
            msg['info/epoch'] = epoch + 1
            msg['info/lr'] = get_lr(optimizer)
            cal_train_metrics(args, msg, outs, labels, batch_size)
            wandb.log(msg)

        train_progress = (batch_id + 1) / total_batchs
        # print(train_progress, show_progress[progress_i])
        if train_progress > show_progress[progress_i]:
            print(".."+str(int(show_progress[progress_i] * 100)) + "%", end='', flush=True)
            progress_i += 1


def main(args, tlogger):
    """
    save model last.pt and best.pt
    """

    train_loader, val_loader, model, optimizer, schedule, scaler, amp_context, start_epoch = set_environment(args, tlogger)

    best_acc = 0.0
    best_eval_name = "null"

    if args.use_wandb:
        wandb.init(entity=args.wandb_entity,
                   project=args.project_name,
                   name=args.exp_name,
                   config=args)
        wandb.run.summary["best_acc"] = best_acc
        wandb.run.summary["best_eval_name"] = best_eval_name
        wandb.run.summary["best_epoch"] = 0

    for epoch in range(start_epoch, args.max_epochs):

        """
        Train
        """
        if train_loader is not None:
            tlogger.print("Start Training {} Epoch".format(epoch+1))
            train(args, epoch, model, scaler, amp_context, optimizer, schedule, train_loader)
            tlogger.print()
        else:
            from eval import eval_and_save
            eval_and_save(args, model, val_loader)
            break

        eval_freq_schedule(args, epoch)

        model_to_save = model.module if hasattr(model, "module") else model
        checkpoint = {"model": model_to_save.state_dict(), "optimizer": optimizer.state_dict(), "epoch":epoch}
        torch.save(checkpoint, args.save_dir + "backup/last.pt")

        if epoch == 0 or (epoch + 1) % args.eval_freq == 0:
            """
            Evaluation
            """
            acc = -1
            if val_loader is not None:
                tlogger.print("Start Evaluating {} Epoch".format(epoch + 1))
                acc, eval_name, accs = evaluate(args, model, val_loader)
                tlogger.print("....BEST_ACC: {}% ({}%)".format(max(acc, best_acc), acc))
                tlogger.print()

            if args.use_wandb:
                wandb.log(accs)

            if acc > best_acc:
                best_acc = acc
                best_eval_name = eval_name
                torch.save(checkpoint, args.save_dir + "backup/best.pt")
            if args.use_wandb:
                wandb.run.summary["best_acc"] = best_acc
                wandb.run.summary["best_eval_name"] = best_eval_name
                wandb.run.summary["best_epoch"] = epoch + 1


if __name__ == "__main__":

    tlogger = timeLogger()

    tlogger.print("Reading Config...")
    args = get_args()
    assert args.c != "", "Please provide config file (.yaml)"
    load_yaml(args, args.c)
    build_record_folder(args)
    tlogger.print()

    main(args, tlogger)