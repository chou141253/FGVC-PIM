import numpy as np
import torch
import torch.nn.functional as F
from typing import Union


@torch.no_grad()
def cal_train_metrics(args, msg: dict, outs: dict, labels: torch.Tensor, batch_size: int):
    """
    only present top-1 training accuracy
    """

    total_loss = 0.0

    if args.use_fpn:
        for i in range(1, 5):
            acc = top_k_corrects(outs["layer"+str(i)].mean(1), labels, tops=[1])["top-1"] / batch_size
            acc = round(acc * 100, 2)
            msg["train_acc/layer{}_acc".format(i)] = acc
            loss = F.cross_entropy(outs["layer"+str(i)].mean(1), labels)
            msg["train_loss/layer{}_loss".format(i)] = loss.item()
            total_loss += loss.item()

    if args.use_selection:
        for name in outs:
            if "select_" not in name:
                continue
            B, S, _ = outs[name].size()
            logit = outs[name].view(-1, args.num_classes)
            labels_0 = labels.unsqueeze(1).repeat(1, S).flatten(0)
            acc = top_k_corrects(logit, labels_0, tops=[1])["top-1"] / (B*S)
            acc = round(acc * 100, 2)
            msg["train_acc/{}_acc".format(name)] = acc
            labels_0 = torch.zeros([B * S, args.num_classes]) - 1
            labels_0 = labels_0.to(args.device)
            loss = F.mse_loss(F.tanh(logit), labels_0)
            msg["train_loss/{}_loss".format(name)] = loss.item()
            total_loss += loss.item()

        for name in outs:
            if "drop_" not in name:
                continue
            B, S, _ = outs[name].size()
            logit = outs[name].view(-1, args.num_classes)
            labels_1 = labels.unsqueeze(1).repeat(1, S).flatten(0)
            acc = top_k_corrects(logit, labels_1, tops=[1])["top-1"] / (B*S)
            acc = round(acc * 100, 2)
            msg["train_acc/{}_acc".format(name)] = acc
            loss = F.cross_entropy(logit, labels_1)
            msg["train_loss/{}_loss".format(name)] = loss.item()
            total_loss += loss.item()

    if args.use_combiner:
        acc = top_k_corrects(outs['comb_outs'], labels, tops=[1])["top-1"] / batch_size
        acc = round(acc * 100, 2)
        msg["train_acc/combiner_acc"] = acc
        loss = F.cross_entropy(outs['comb_outs'], labels)
        msg["train_loss/combiner_loss"] = loss.item()
        total_loss += loss.item()

    if "ori_out" in outs:
        acc = top_k_corrects(outs["ori_out"], labels, tops=[1])["top-1"] / batch_size
        acc = round(acc * 100, 2)
        msg["train_acc/ori_acc"] = acc
        loss = F.cross_entropy(outs["ori_out"], labels)
        msg["train_loss/ori_loss"] = loss.item()
        total_loss += loss.item()

    msg["train_loss/total_loss"] = total_loss



@torch.no_grad()
def top_k_corrects(preds: torch.Tensor, labels: torch.Tensor, tops: list = [1, 3, 5]):
    """
    preds: [B, C] (C is num_classes)
    labels: [B, ]
    """
    if preds.device != torch.device('cpu'):
        preds = preds.cpu()
    if labels.device != torch.device('cpu'):
        labels = labels.cpu()
    tmp_cor = 0
    corrects = {"top-"+str(x):0 for x in tops}
    sorted_preds = torch.sort(preds, dim=-1, descending=True)[1]
    for i in range(tops[-1]):
        tmp_cor += sorted_preds[:, i].eq(labels).sum().item()
        # records
        if "top-"+str(i+1) in corrects:
            corrects["top-"+str(i+1)] = tmp_cor
    return corrects


@torch.no_grad()
def _cal_evalute_metric(corrects: dict, 
                        total_samples: dict,
                        logits: torch.Tensor, 
                        labels: torch.Tensor, 
                        this_name: str,
                        scores: Union[list, None] = None, 
                        score_names: Union[list, None] = None):
    
    tmp_score = torch.softmax(logits, dim=-1)
    tmp_corrects = top_k_corrects(tmp_score, labels, tops=[1, 3]) # return top-1, top-3, top-5 accuracy
    
    ### each layer's top-1, top-3 accuracy
    for name in tmp_corrects:
        eval_name = this_name + "-" + name
        if eval_name not in corrects:
            corrects[eval_name] = 0
            total_samples[eval_name] = 0
        corrects[eval_name] += tmp_corrects[name]
        total_samples[eval_name] += labels.size(0)
    
    if scores is not None:
        scores.append(tmp_score)
    if score_names is not None:
        score_names.append(this_name)


@torch.no_grad()
def _average_top_k_result(corrects: dict, total_samples: dict, scores: list, labels: torch.Tensor, tops: list = [1, 2, 3, 4, 5]):
    """
    scores is a list contain:
    [
        tensor1, 
        tensor2,...
    ] tensor1 and tensor2 have same size [B, num_classes]
    """
    # initial
    for t in tops:
        eval_name = "highest-{}".format(t)
        if eval_name not in corrects:
            corrects[eval_name] = 0
            total_samples[eval_name] = 0
        total_samples[eval_name] += labels.size(0)

    if labels.device != torch.device('cpu'):
        labels = labels.cpu()
    
    batch_size = labels.size(0)
    scores_t = torch.cat([s.unsqueeze(1) for s in scores], dim=1) # B, 5, C

    if scores_t.device != torch.device('cpu'):
        scores_t = scores_t.cpu()

    max_scores = torch.max(scores_t, dim=-1)[0]
    # sorted_ids = torch.sort(max_scores, dim=-1, descending=True)[1] # this id represents different layers outputs, not samples

    for b in range(batch_size):
        tmp_logit = None
        ids = torch.sort(max_scores[b], dim=-1)[1] # S
        for i in range(tops[-1]):
            top_i_id = ids[i]
            if tmp_logit is None:
                tmp_logit = scores_t[b][top_i_id]
            else:
                tmp_logit += scores_t[b][top_i_id]
            # record results
            if i+1 in tops:
                if torch.max(tmp_logit, dim=-1)[1] == labels[b]:
                    eval_name = "highest-{}".format(i+1)
                    corrects[eval_name] += 1


def evaluate(args, model, test_loader):
    """
    [Notice: Costom Model]
    If you use costom model, please change fpn module return name (under 
    if args.use_fpn: ...)
    [Evaluation Metrics]
    We calculate each layers accuracy, combiner accuracy and average-higest-1 ~ 
    average-higest-5 accuracy (average-higest-5 means average all predict scores
    as final predict)
    """

    model.eval()
    corrects = {}
    total_samples = {}

    total_batchs = len(test_loader) # just for log
    show_progress = [x/10 for x in range(11)] # just for log
    progress_i = 0

    with torch.no_grad():
        """ accumulate """
        for batch_id, (ids, datas, labels) in enumerate(test_loader):
            
            score_names = []
            scores = []
            datas = datas.to(args.device)

            outs = model(datas)

            if args.use_fpn:
                for i in range(1, 5):
                    this_name = "layer" + str(i)
                    _cal_evalute_metric(corrects, total_samples, outs[this_name].mean(1), labels, this_name, scores, score_names)
            
            ### for research
            if args.use_selection:
                for name in outs:
                    if "select_" not in name:
                        continue
                    this_name = name
                    S = outs[name].size(1)
                    logit = outs[name].view(-1, args.num_classes)
                    labels_1 = labels.unsqueeze(1).repeat(1, S).flatten(0)
                    _cal_evalute_metric(corrects, total_samples, logit, labels_1, this_name)
                
                for name in outs:
                    if "drop_" not in name:
                        continue
                    this_name = name
                    S = outs[name].size(1)
                    logit = outs[name].view(-1, args.num_classes)
                    labels_0 = labels.unsqueeze(1).repeat(1, S).flatten(0)
                    _cal_evalute_metric(corrects, total_samples, logit, labels_0, this_name)

            if args.use_combiner:
                this_name = "combiner"
                _cal_evalute_metric(corrects, total_samples, outs["comb_outs"], labels, this_name, scores, score_names)

            if "ori_out" in outs:
                this_name = "original"
                _cal_evalute_metric(corrects, total_samples, outs["ori_out"], labels, this_name)
        
            _average_top_k_result(corrects, total_samples, scores, labels)

            eval_progress = (batch_id + 1) / total_batchs
            
            if eval_progress > show_progress[progress_i]:
                print(".."+str(int(show_progress[progress_i]*100))+"%", end='', flush=True)
                progress_i += 1

        """ calculate accuracy """
        # total_samples = len(test_loader.dataset)
        
        best_top1 = 0.0
        best_top1_name = ""
        eval_acces = {}
        for name in corrects:
            acc = corrects[name] / total_samples[name]
            acc = round(100 * acc, 3)
            eval_acces[name] = acc
            ### only compare top-1 accuracy
            if "top-1" in name or "highest" in name:
                if acc >= best_top1:
                    best_top1 = acc
                    best_top1_name = name

    return best_top1, best_top1_name, eval_acces


@torch.no_grad()
def eval_and_save(args, model, val_loader):
    tlogger.print("Start Evaluating")
    acc, eval_name, accs = evaluate(args, model, val_loader)
    tlogger.print("....BEST_ACC: {} {}%".format(eval_name, acc))
    ### build records.txt
    msg = "[Evaluation Results]\n"
    msg += "Project: {}, Experiment: {}\n".format(args.project_name, args.exp_name)
    msg += "Samples: {}\n".format(len(val_loader.dataset))
    msg += "\n"
    for name in eval_acces:
        msg += "    {} {}%\n".format(name, eval_acces[name])
    msg += "\n"
    msg += "BEST_ACC: {} {}% ".format(eval_name, acc)
    
    with open(args.save_dir + "eval_results.txt", "w") as ftxt:
        ftxt.write(msg)

