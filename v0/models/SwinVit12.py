import torch
import torch.nn as nn
import torchvision.models as models
import timm
import math
from scipy import ndimage
import numpy as np
import copy

def load_model_weights(model, model_path):
    state = torch.load(model_path, map_location='cpu')
    for key in model.state_dict():
        if 'num_batches_tracked' in key:
            continue
        p = model.state_dict()[key]
        if key in state['state_dict']:
            ip = state['state_dict'][key]
            if p.shape == ip.shape:
                p.data.copy_(ip.data)  # Copy the data of parameters
            else:
                print('could not load layer: {}, mismatch shape {} ,{}'.format(key, (p.shape), (ip.shape)))
        else:
            print('could not load layer: {}, not in checkpoint'.format(key))
    return model

class GCN(nn.Module):

    def __init__(self, 
                 num_joints: int, 
                 in_features: int, 
                 num_classes: int, 
                 use_global_token: bool = False):
        super(GCN, self).__init__()
        
        joints = [num_joints//32, num_joints//32, num_joints//16]

        # 1
        self.pool1 = nn.Linear(num_joints, joints[0])

        A = torch.eye(joints[0])/100 + 1/100
        self.adj1 = nn.Parameter(copy.deepcopy(A))
        self.conv1 = nn.Conv1d(in_features, in_features, 1)
        self.batch_norm1 = nn.BatchNorm1d(in_features)
        
        self.conv_q1 = nn.Conv1d(in_features, in_features//4, 1)
        self.conv_k1 = nn.Conv1d(in_features, in_features//4, 1)
        self.alpha1 = nn.Parameter(torch.zeros(1))

        # # 2
        # self.pool2 = nn.Linear(joints[0], joints[1])

        # A = torch.eye(joints[1])/100 + 1/100
        # self.adj2 = nn.Parameter(copy.deepcopy(A))
        # self.conv2 = nn.Conv1d(in_features, in_features, 1)
        # self.batch_norm2 = nn.BatchNorm1d(in_features)
        
        # self.conv_q2 = nn.Conv1d(in_features, in_features//4, 1)
        # self.conv_k2 = nn.Conv1d(in_features, in_features//4, 1)
        # self.alpha2 = nn.Parameter(torch.zeros(1))

        # # 3
        # self.pool3 = nn.Linear(joints[1], joints[2])
        
        # A = torch.eye(joints[2])/100 + 1/100
        # self.adj3 = nn.Parameter(copy.deepcopy(A))
        # self.conv3 = nn.Conv1d(in_features, in_features, 1)
        # self.batch_norm3 = nn.BatchNorm1d(in_features)
        
        # self.conv_q3 = nn.Conv1d(in_features, in_features//4, 1)
        # self.conv_k3 = nn.Conv1d(in_features, in_features//4, 1)
        # self.alpha3 = nn.Parameter(torch.zeros(1))

        self.pool4 = nn.Linear(joints[0], 1)
        
        self.dropout = nn.Dropout(p=0.1)
        self.classifier = nn.Linear(in_features, num_classes)

        self.tanh = nn.Tanh()
        

    def forward(self, x):

        x = self.pool1(x)
        q1 = self.conv_q1(x).mean(1)
        k1 = self.conv_k1(x).mean(1)
        A1 = self.tanh(q1.unsqueeze(-1) - k1.unsqueeze(1))
        A1 = self.adj1 + A1 * self.alpha1
        x = self.conv1(x)
        x = torch.matmul(x, A1)
        x = self.batch_norm1(x)

        # x = self.pool2(x)
        # q2 = self.conv_q2(x).mean(1)
        # k2 = self.conv_k2(x).mean(1)
        # A2 = self.tanh(q2.unsqueeze(-1) - k2.unsqueeze(1))
        # A2 = self.adj2 + A2 * self.alpha2
        # x = self.conv2(x)
        # x = torch.matmul(x, A2)
        # x = self.batch_norm2(x)

        # x = self.pool3(x)
        # q3 = self.conv_q3(x).mean(1)
        # k3 = self.conv_k3(x).mean(1)
        # A3 = self.tanh(q3.unsqueeze(-1) - k3.unsqueeze(1))
        # A3 = self.adj3 + A3 * self.alpha3
        # x = self.conv3(x)
        # x = torch.matmul(x, A3)
        # x = self.batch_norm3(x)

        x = self.pool4(x)
        x = self.dropout(x)
        x = x.flatten(1)
        x = self.classifier(x)
        
        return x



# class GCN(nn.Module):

#     def __init__(self, 
#                  num_joints: int, 
#                  in_features: int, 
#                  num_classes: int,
#                  use_global_token: bool = False):
#         super(GCN, self).__init__()
#         self.num_joints = num_joints
#         self.in_features = in_features
#         self.num_classes = num_classes

#         A = torch.eye(num_joints)/4000 + 1/4000

#         self.batch_norm1 = nn.BatchNorm1d(in_features)

#         self.adj1 = nn.Parameter(copy.deepcopy(A))

#         self.conv1 = nn.Conv1d(in_features, in_features, 1)
        
#         self.conv_q = nn.Conv1d(in_features, in_features//4, 1)
#         self.conv_k = nn.Conv1d(in_features, in_features//4, 1)

#         self.alpha = nn.Parameter(torch.zeros(1))

#         # self.adj2 = nn.Parameter(A)

#         # self.conv1 = nn.Sequential(
#         #     nn.Conv1d(in_features, 4*in_features, 1),
#         #     nn.Dropout(p=0.2),
#         #     nn.GELU(),
#         #     nn.Conv1d(4*in_features, in_features, 1),
#         # )

#         # self.conv2 = nn.Sequential(
#         #     nn.Conv1d(in_features, 4*in_features, 1),
#         #     nn.Dropout(p=0.2),
#         #     nn.GELU(),
#         #     nn.Conv1d(4*in_features, in_features, 1),
#         # )
        
#         self.dropout = nn.Dropout(p=0.1)
#         self.classifier = nn.Conv1d(in_features, num_classes, 1)

#         self.avgpool = nn.AdaptiveAvgPool1d((1))

#         self.tanh = nn.Tanh()

#     def forward(self, x):
#         """
#         x size: [B, C, N]
#         """

#         q = self.conv_q(x).mean(1)
#         k = self.conv_k(x).mean(1)
#         A = self.tanh(q.unsqueeze(-1) - k.unsqueeze(1))
#         A = self.adj1 + A * self.alpha
#         x = self.conv1(x)
#         x = torch.matmul(x, A)
#         x = self.batch_norm1(x)
        
#         x = self.dropout(x)
#         x = self.avgpool(x)
#         x = self.classifier(x)
        
#         return x

class SwinVit12(nn.Module):

    def __init__(self, 
                 in_size: int,
                 num_classes: int, 
                 use_fpn: bool, 
                 use_ori: bool, 
                 use_gcn: bool, 
                 use_layers: list,
                 use_selections: list,
                 num_selects: list,
                 global_feature_dim: int = 2048):
        super(SwinVit12, self).__init__()
        """
        (features)
            (0)~(8)
        (avgpool)
        (classifier)
        
        use_layers : about classifier prediction loss.
        use_selection : about select prediction loss.
        """
        # BNC
        self.in_size = in_size
        self.layer_dims = [[2304, 384],
                           [576, 768],
                           [144, 1536],
                           [144, 1536]]

        self.num_layers = len(self.layer_dims)
        
        self.total_pathes = (in_size//16)**2 + 1

        # assert len(use_layers) == self.num_layers
        # assert len(use_selections) == len(use_layers)
        self.num_classes = num_classes
        self.num_selects = num_selects
        self.check_input(use_layers, use_selections) # layer --> selection
        self.use_layers = use_layers
        self.use_selections = use_selections
        self.global_feature_dim = global_feature_dim
        self.use_fpn = use_fpn
        self.use_ori = use_ori
        self.use_gcn = use_gcn

        # create features extractor
        # test1 'swin_large_patch4_window12_384_in22k'
        # test2 'swin_large_patch4_window7_224_in22k'
        self.extractor = timm.create_model('swin_large_patch4_window12_384_in22k', pretrained=True)
        # self.extractor = load_model_weights(self.extractor, "./models/vit_base_patch16_224_miil_21k.pth")
        # with open("structure.txt", "w") as ftxt:
        #     ftxt.write(str(self.extractor))

        self.only_ori = use_ori and not (use_fpn or use_gcn)
        
        if self.only_ori:
            self.extractor.head = nn.Sequential(
                nn.Linear(global_feature_dim, global_feature_dim),
                nn.ReLU(),
                nn.Dropout(p=0.1),
                nn.Linear(global_feature_dim, num_classes)
            )
        elif use_ori:
            self.extractor.head = nn.Linear(global_feature_dim, num_classes)

        print(str(self.extractor))

        # fpn module
        if use_fpn:
            for i in range(self.num_layers):
                self.add_module("fpn_"+str(i), 
                                nn.Sequential(
                                    nn.Linear(self.layer_dims[i][1], self.layer_dims[i][1]),
                                    nn.ReLU(),
                                    nn.Linear(self.layer_dims[i][1], global_feature_dim)
                                ))

                if i != 0:
                    if self.layer_dims[i][0] != self.layer_dims[i-1][0]:
                        self.add_module("upsample_"+str(i), 
                                            nn.Conv1d(self.layer_dims[i][0], self.layer_dims[i-1][0], 1)
                                        )

        # mlp classifier (layer classifier module).
        for i in range(self.num_layers):

            if use_layers[i] and not use_fpn:
                self.add_module("proj_l"+str(i),
                    nn.Conv2d(self.layer_dims[i][1], global_feature_dim, 1))

            if use_layers[i]:
                self.add_module("classifier_l"+str(i),
                    nn.Sequential(
                            nn.Conv2d(global_feature_dim, global_feature_dim, 1),
                            nn.BatchNorm2d(global_feature_dim),
                            nn.ReLU(),
                            nn.Conv2d(global_feature_dim, num_classes, 1)
                        ))

        if self.use_gcn:
            num_joints = 0 # without global token.
            for n in self.num_selects:
                if n != 0:
                    num_joints += n

            self.gcn = GCN(num_joints = num_joints, 
                           in_features = global_feature_dim, 
                           num_classes = num_classes)

        self.crossentropy = nn.CrossEntropyLoss()
        self.bce = nn.BCEWithLogitsLoss()
        self.mseloss = nn.MSELoss()
        self.tanh = nn.Tanh()

    def check_input(self, use_layers: list, use_selections: list):
        for i in range(len(use_layers)):
            if use_selections[i] and not use_layers[i]:
                raise ValueError("selection loss after layer loss.")

    @torch.no_grad()
    def _accuracy(self, logits, labels):
        _, indices = torch.max(logits.detach().cpu(), dim=-1)
        corrects = indices.eq(labels.detach().cpu()).sum().item()
        acc = corrects / labels.size(0)
        return acc

    @torch.no_grad()
    def _selected_accuracy(self, selected_logits, labels, best_5=True):
        """
        selected_features: B, S, C
        """
        labels = labels.detach().cpu()
        logits = selected_logits["selected"].detach().cpu()
        B = logits.size(0)
        corrects = 0
        total = 0
        for i in range(B):
            _, indices = torch.max(logits[i], dim=-1)
            corrects += (indices == labels[i]).sum().item()
            total += indices.size(0)
        selected_acc = corrects / total

        logits = selected_logits["not_selected"].detach().cpu()
        B = logits.size(0)
        corrects = 0
        for i in range(B):
            _, indices = torch.max(logits[i], dim=-1)
            corrects += (indices == labels[i]).sum().item()
            total += indices.size(0)
        not_selected_acc = corrects / total

        return [selected_acc, not_selected_acc]

    def _loss(self, logits, labels, merge=True):
        B, C, H, W = logits.size()
        if merge:
            logits = logits.view(B, C, -1).transpose(2, 1).contiguous().mean(dim=1)
        else:
            logits = logits.view(B, C, -1).transpose(2, 1).contiguous().view(-1, C)
            S = int(H*W)
            labels = labels.unsqueeze(1).repeat(1, S).flatten()

        loss = self.crossentropy(logits, labels)
        acc = self._accuracy(logits, labels)
        return 0.5*loss, acc
    
    def _select_loss(self, selected_logits, labels):
        loss1 = 0
        # logits1 = selected_logits["selected"]
        # B, S1, C = logits1.size()
        # logits1 = logits1.view(-1, C)
        # labels1 = labels.unsqueeze(1).repeat(1, S1).flatten()
        # loss1 = self.crossentropy(logits1, labels1)
        
        logits2 = selected_logits["not_selected"]
        B, S2, C = logits2.size()
        logits2 = logits2.view(-1, C)
        logits2 = self.tanh(logits2)
        labels2 = torch.zeros([B*S2, C]) - 1
        labels2 = labels2.to(logits2.device)
        loss2 = 5*self.mseloss(logits2, labels2)

        return loss1, loss2

    def _select_features(self, logits, features, num_select):
        # prepare, B, C, H, W --> B, S, C
        selected_logits = {"selected":[], "not_selected":[]}
        B, C, H, W = logits.size()
        logits = logits.view(B, C, -1).transpose(2, 1).contiguous()
        B, C, H, W = features.size()
        features = features.view(B, C, -1).transpose(2, 1).contiguous()
        # measure probabilitys.
        probs = torch.softmax(logits, dim=-1)
        selected_features = []
        selected_confs = []
        for bi in range(B):
            max_ids, _ = torch.max(probs[bi], dim=-1)
            confs, ranks = torch.sort(max_ids, descending=True)
            sf = features[bi][ranks[:num_select]]
            nf = features[bi][ranks[num_select:]]  # calculate
            selected_features.append(sf) # [num_selected, C]
            selected_confs.append(confs) # [num_selected]
            selected_logits["selected"].append(logits[bi][ranks[:num_select]])
            selected_logits["not_selected"].append(logits[bi][ranks[num_select:]])
        
        selected_features = torch.stack(selected_features)
        selected_confs = torch.stack(selected_confs)
        selected_logits["selected"] = torch.stack(selected_logits["selected"])
        selected_logits["not_selected"] = torch.stack(selected_logits["not_selected"])
        
        return selected_features, selected_confs, selected_logits

    def _upsample_add(self, x1, x0):

        B, L, C = x1.shape
        S = int(L**0.5)
        x1 = x1.permute(0, 2, 1).contiguous().view(B, C, S, S)
        x1 = self.upsample(x1)

        B, L, C = x0.shape
        S = int(L**0.5)
        x0 = x0.permute(0, 2, 1).contiguous().view(B, C, S, S)
        x = x1 + x0
        x = x.view(B, C, -1).permute(0, 2, 1).contiguous()

        return x

    def forward(self, x, labels, return_preds=False):
        """
        x: [B, C, H, W]
        labels: [B]

        compute

        original prediction ---> loss_ori.
        layers prediction ---> loss_layer.
        layer selections ---> selected loss and not selected loss.
        """

        # = = = = = restore predictions. = = = = =
        logits = {}
        # confidences = {}
        accuracys = {}
        losses = {}
        selected_features = []

        batch_size = x.size(0)
        
        layers = list(self.extractor(x))

        # = = = = = fpn = = = = =
        if self.use_fpn:
            layers[-1] = getattr(self, "fpn_"+str(len(layers)-1))(layers[-1])
            for i in range(self.num_layers-1, 0, -1):
                if self.layer_dims[i][0] != self.layer_dims[i-1][0]:
                    layers[i-1] = getattr(self, "fpn_"+str(i-1))(layers[i-1]) + getattr(self, "upsample_"+str(i))(layers[i])
                else:
                    layers[i-1] = getattr(self, "fpn_"+str(i-1))(layers[i-1]) + layers[i]
        
        # layers prediction
        for i in range(self.num_layers):
            
            if self.use_layers[i]:

                B, L, C = layers[i].shape
                S = int(L**0.5)
                layers[i] = layers[i].transpose(2, 1).contiguous().view(B, C, S, S)

                if not self.use_fpn:
                    layers[i] = getattr(self, "proj_l"+str(i))(layers[i])
                
                # layer predictions.
                logits["l_"+str(i)] = getattr(self, "classifier_l"+str(i))(layers[i])

                # select features.
                if self.use_selections[i]:
                    sf, sc, sl = self._select_features(logits=logits["l_"+str(i)], 
                                                       features=layers[i],
                                                       num_select=self.num_selects[i])

                    if self.use_gcn:
                        selected_features.append(sf) # restore selected features.

                    # compute selected loss.
                    l_loss_s1,  l_loss_s2 = self._select_loss(sl, labels)
                    losses["l"+str(i)+"_selected"] = l_loss_s1
                    losses["l"+str(i)+"_not_selected"] = l_loss_s2
                    # compute selected accuracy.
                    acc1, acc2 = self._selected_accuracy(sl, labels)
                    accuracys["l"+str(i)+"_selected"] = acc1
                    accuracys["l"+str(i)+"_not_selected"] = acc2
                    
                    logits["l"+str(i)+"_selected"] = sl["selected"]
                    # confidences["l"+str(i)+"_selected"] = sc
                
        # original prediction.
        
        if self.use_ori:
            if not self.only_ori:
                B, C, S, S = layers[-1].shape
                layers[-1] = layers[-1].view(B, C, -1).transpose(1, 2).contiguous()
            ori_x = self.extractor.norm(layers[-1])  # B L C
            ori_x = self.extractor.avgpool(ori_x.transpose(1, 2))  # B C 1
            ori_x = torch.flatten(ori_x, 1)
            logits["ori"] = self.extractor.head(ori_x)
            losses["ori"] = self.crossentropy(logits["ori"], labels)
            accuracys["ori"] = self._accuracy(logits["ori"], labels)

        # selected prediction.
        if self.use_gcn:
            selected_features = torch.cat(selected_features, dim=1) # B, S, C
            selected_features = selected_features.transpose(1, 2).contiguous()
            logits["gcn"] = self.gcn(selected_features)
            losses["gcn"] = self.crossentropy(logits["gcn"], labels)
            accuracys["gcn"] = self._accuracy(logits["gcn"], labels)

        for i in range(self.num_layers):
            if self.use_layers[i]:
                loss, acc = self._loss(logits["l_"+str(i)], labels)
                losses["l_"+str(i)] = loss
                accuracys["l_"+str(i)] = acc

        # return
        if return_preds:
            return losses, accuracys, logits

        return losses, accuracys