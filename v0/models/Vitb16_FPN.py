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

        joints = [num_joints//8, num_joints//16, num_joints//32]

        # 1
        self.pool1 = nn.Linear(num_joints, joints[0])

        A = torch.eye(joints[0])/100 + 1/100
        self.adj1 = nn.Parameter(copy.deepcopy(A))
        self.conv1 = nn.Conv1d(in_features, in_features, 1)
        self.batch_norm1 = nn.BatchNorm1d(in_features)
        
        self.conv_q1 = nn.Conv1d(in_features, in_features//4, 1)
        self.conv_k1 = nn.Conv1d(in_features, in_features//4, 1)
        self.alpha1 = nn.Parameter(torch.zeros(1))

        # 2
        self.pool2 = nn.Linear(joints[0], joints[1])

        A = torch.eye(joints[1])/32 + 1/32
        self.adj2 = nn.Parameter(copy.deepcopy(A))
        self.conv2 = nn.Conv1d(in_features, in_features, 1)
        self.batch_norm2 = nn.BatchNorm1d(in_features)
        
        self.conv_q2 = nn.Conv1d(in_features, in_features//4, 1)
        self.conv_k2 = nn.Conv1d(in_features, in_features//4, 1)
        self.alpha2 = nn.Parameter(torch.zeros(1))

        # 3
        self.pool3 = nn.Linear(joints[1], joints[2])

        A = torch.eye(joints[2])/32 + 1/32
        self.adj3 = nn.Parameter(copy.deepcopy(A))
        self.conv3 = nn.Conv1d(in_features, in_features, 1)
        self.batch_norm3 = nn.BatchNorm1d(in_features)
        
        self.conv_q3 = nn.Conv1d(in_features, in_features//4, 1)
        self.conv_k3 = nn.Conv1d(in_features, in_features//4, 1)
        self.alpha3 = nn.Parameter(torch.zeros(1))

        self.pool4 = nn.Linear(joints[2], 1)
        
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

        x = self.pool2(x)
        q2 = self.conv_q2(x).mean(1)
        k2 = self.conv_k2(x).mean(1)
        A2 = self.tanh(q2.unsqueeze(-1) - k2.unsqueeze(1))
        A2 = self.adj2 + A2 * self.alpha2
        x = self.conv2(x)
        x = torch.matmul(x, A2)
        x = self.batch_norm2(x)

        x = self.pool3(x)
        q3 = self.conv_q3(x).mean(1)
        k3 = self.conv_k3(x).mean(1)
        A3 = self.tanh(q3.unsqueeze(-1) - k3.unsqueeze(1))
        A3 = self.adj3 + A3 * self.alpha3
        x = self.conv2(x)
        x = torch.matmul(x, A3)
        x = self.batch_norm2(x)

        x = self.pool4(x)
        x = self.dropout(x)
        x = x.flatten(1)
        x = self.classifier(x)

        return x
class VitB16(nn.Module):

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
        super(VitB16, self).__init__()
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
        self.layer_dims = [[(in_size//16)**2, 768],
                           [(in_size//16)**2, 768],
                           [(in_size//16)**2, 768],
                           [(in_size//16)**2, 768],
                           [(in_size//16)**2, 768],
                           [(in_size//16)**2, 768],
                           [(in_size//16)**2, 768],
                           [(in_size//16)**2, 768],
                           [(in_size//16)**2, 768],
                           [(in_size//16)**2, 768],
                           [(in_size//16)**2, 768],
                           [(in_size//16)**2, 768]]

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
        self.extractor = timm.create_model('vit_base_patch16_224_miil_in21k', pretrained=False)
        self.extractor = load_model_weights(self.extractor, "./models/vit_base_patch16_224_miil_21k.pth")
        
        # different length
        # trans-fg
        # thanks: https://github.com/TACJu/TransFG/blob/master/models/modeling.py

        posemb_tok, posemb_grid = self.extractor.pos_embed[:, :1], self.extractor.pos_embed[0, 1:]
        posemb_grid = posemb_grid.detach().numpy()
        gs_old = int(math.sqrt(len(posemb_grid)))
        gs_new = in_size//16
        print('load_pretrained: grid-size from %s to %s' % (gs_old, gs_new))
        posemb_grid = posemb_grid.reshape(gs_old, gs_old, -1)
        zoom = (gs_new / gs_old, gs_new / gs_old, 1)
        posemb_grid = ndimage.zoom(posemb_grid, zoom, order=1)
        posemb_grid = posemb_grid.reshape(1, gs_new * gs_new, -1)
        posemb_grid = torch.from_numpy(posemb_grid)
        # print(posemb_tok.size(), posemb_grid.size())
        posemb = torch.cat([posemb_tok, posemb_grid], dim=1)
        self.extractor.pos_embed = torch.nn.Parameter(posemb)

        if use_ori:
            self.extractor.head = nn.Sequential(
                nn.Linear(global_feature_dim, global_feature_dim),
                nn.ReLU(),
                nn.Dropout(p=0.1),
                nn.Linear(global_feature_dim, num_classes)
            )

        # fpn module
        if use_fpn:
            for i in range(self.num_layers):
                self.add_module("fpn_"+str(i), 
                                nn.Sequential(
                                    nn.Conv1d(self.layer_dims[i][1], global_feature_dim, 1),
                                    nn.ReLU(),
                                    nn.Conv1d(global_feature_dim, global_feature_dim, 1),
                                ))
                self.add_module("upsample_"+str(i), 
                                 nn.Linear(self.layer_dims[i][0]+1, self.layer_dims[i][0]+1))

        # mlp classifier (layer classifier module).
        for i in range(self.num_layers):

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
                    num_joints += n + 1

            self.gcn = GCN(num_joints = num_joints, 
                           in_features = global_feature_dim, 
                           num_classes = num_classes)

        # self.upsample = nn.Upsample(scale_factor=2, mode='bilinear')
        self.crossentropy = nn.CrossEntropyLoss()
        self.bce = nn.BCEWithLogitsLoss()

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

    def _loss(self, logits, labels):
        B, C, H, W = logits.size()
        logits = logits.view(B, C, -1).transpose(2, 1).contiguous().mean(dim=1)
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
        labels2 = torch.zeros([B*S2, C])
        labels2 = labels2.to(logits2.device)
        loss2 = self.bce(logits2, labels2)
        
        return loss1, 5*loss2

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
        x1 = self.upsample(x1)
        return x1 + x0

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
        # position embedding and global patch.
        x = self.extractor.patch_embed(x) # x input size [224, 224]
        cls_token = self.extractor.cls_token.expand(x.shape[0], -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_token, x), dim=1)
        x = self.extractor.pos_drop(x + self.extractor.pos_embed)

        # blocks
        layers = [self.extractor.blocks[0](x)]
        for i in range(1, self.num_layers):
            layers.append(self.extractor.blocks[i](layers[-1]))

        for i in range(self.num_layers):
            layers[i] = layers[i].transpose(2, 1).contiguous() # BNC --> BCHw


        # = = = = = fpn = = = = =
        if self.use_fpn:
            layers[-1] = getattr(self, "fpn_"+str(len(layers)-1))(layers[-1])
            for i in range(self.num_layers-1, 0, -1):
                layers[i-1] = getattr(self, "fpn_"+str(i-1))(layers[i-1]) + getattr(self, "upsample_"+str(i))(layers[i])

        
        # layers prediction
        for i in range(self.num_layers):

            layers[i] = layers[i].unsqueeze(-1)
            
            if self.use_layers[i]:

                # if not self.use_fpn:
                #     layers[i] = getattr(self, "proj_l"+str(i))(layers[i])
                
                # layer predictions.
                logits["l_"+str(i)] = getattr(self, "classifier_l"+str(i))(layers[i])

                # select features.
                if self.use_selections[i]:
                    sf, sc, sl = self._select_features(logits=logits["l_"+str(i)][:, :, 1:], 
                                                       features=layers[i],
                                                       num_select=self.num_selects[i])

                    if self.use_gcn:
                        selected_features.append(layers[i][:, :, 0, 0].unsqueeze(1))
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
            logits["ori"] = self.extractor.head(layers[-1][:, :, 0, 0])
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