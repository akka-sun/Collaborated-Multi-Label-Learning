import torch
from torch import nn as nn, Tensor
import os
import pandas as pd
import numpy as np


class PartialSelectiveLoss(nn.Module):

    def __init__(self):
        super(PartialSelectiveLoss, self).__init__()
        self.clip = 0
        self.gamma_pos = 0
        self.gamma_neg = 1
        self.gamma_unann = 4
        self.alpha_pos = 1
        self.alpha_neg = 1
        self.alpha_unann = 1

        self.targets_weights = None
        print("Prior file was found in given path.")
        df = pd.read_csv('/root/autodl-tmp/muti_label_classify/PartialLabelingCSL/outputs/priors/prior_fpc_1000.csv')
        self.prior_classes = dict(zip(df.values[:, 0], df.values[:, 1]))
        print("Prior file was loaded successfully. ")

    def forward(self, logits, targets,idx):

        # Positive, Negative and Un-annotated indexes
        targets_pos = (targets == 1).float()
        targets_neg = (targets == 0).float()
        targets_unann = (idx == 0).float()
        targets_neg=targets_neg-targets_unann
        # print(targets_neg)
        # print(targets_unann)

        # Activation
        xs_pos = torch.sigmoid(logits)
        xs_neg = 1.0 - xs_pos

        if self.clip is not None and self.clip > 0:
            xs_neg.add_(self.clip).clamp_(max=1)

        prior_classes = None
        if hasattr(self, "prior_classes"):
            prior_classes = torch.tensor(list(self.prior_classes.values())).cuda()

        targets_weights = self.targets_weights
        targets_weights, xs_neg = edit_targets_parital_labels( targets, targets_weights, xs_neg,
                                                              prior_classes=prior_classes)

        # Loss calculation
        BCE_pos = self.alpha_pos * targets_pos * torch.log(torch.clamp(xs_pos, min=1e-8))
        BCE_neg = self.alpha_neg * targets_neg * torch.log(torch.clamp(xs_neg, min=1e-8))
        BCE_unann = self.alpha_unann * targets_unann * torch.log(torch.clamp(xs_neg, min=1e-8))

        BCE_loss = BCE_pos + BCE_neg + BCE_unann

        # Adding asymmetric gamma weights
        with torch.no_grad():
            asymmetric_w = torch.pow(1 - xs_pos * targets_pos - xs_neg * (targets_neg + targets_unann),
                                     self.gamma_pos * targets_pos + self.gamma_neg * targets_neg +
                                     self.gamma_unann * targets_unann)
        BCE_loss *= asymmetric_w

        # partial labels weights
        BCE_loss *= targets_weights

        return -BCE_loss.sum()


def edit_targets_parital_labels( targets, targets_weights, xs_neg, prior_classes=None):
    likelihood_topk = 5
    if targets_weights is None or targets_weights.shape != targets.shape:
        targets_weights = torch.ones(targets.shape, device=torch.device('cuda'))
    else:
        targets_weights[:] = 1.0
    num_top_k = likelihood_topk * targets_weights.shape[0]

    xs_neg_prob = xs_neg
    prior_threshold = 0.5
    if prior_classes is not None:
        if prior_threshold:
            idx_ignore = torch.where(prior_classes > prior_threshold)[0]
            targets_weights[:, idx_ignore] = 0
            targets_weights += (targets != -1).float()
            targets_weights = targets_weights.bool()

    negative_backprop_fun_jit(targets, xs_neg_prob, targets_weights, num_top_k)

    # set all unsure targets as negative
    # targets[targets == -1] = 0

    return targets_weights, xs_neg


# @torch.jit.script
def negative_backprop_fun_jit(targets: Tensor, xs_neg_prob: Tensor, targets_weights: Tensor, num_top_k: int):
    with torch.no_grad():
        targets_flatten = targets.flatten()
        cond_flatten = torch.where(targets_flatten == -1)[0]
        targets_weights_flatten = targets_weights.flatten()
        xs_neg_prob_flatten = xs_neg_prob.flatten()
        ind_class_sort = torch.argsort(xs_neg_prob_flatten[cond_flatten])
        targets_weights_flatten[
            cond_flatten[ind_class_sort[:num_top_k]]] = 0


class ComputePrior:
    def __init__(self, classes):
        self.classes = classes
        n_classes = len(self.classes)
        self.sum_pred_train = torch.zeros(n_classes).cuda()
        self.sum_pred_val = torch.zeros(n_classes).cuda()
        self.cnt_samples_train,  self.cnt_samples_val = .0, .0
        self.avg_pred_train, self.avg_pred_val = None, None
        self.path_dest = "./outputs"
        self.path_local = "/class_prior/"

    def update(self, logits, training=True):
        with torch.no_grad():
            preds = torch.sigmoid(logits).detach()
            if training:
                self.sum_pred_train += torch.sum(preds, axis=0)
                self.cnt_samples_train += preds.shape[0]
                self.avg_pred_train = self.sum_pred_train / self.cnt_samples_train

            else:
                self.sum_pred_val += torch.sum(preds, axis=0)
                self.cnt_samples_val += preds.shape[0]
                self.avg_pred_val = self.sum_pred_val / self.cnt_samples_val

    def save_prior(self):

        print('Prior (train), first 5 classes: {}'.format(self.avg_pred_train[:5]))

        # Save data frames as csv files
        if not os.path.exists(self.path_dest):
            os.makedirs(self.path_dest)

        df_train = pd.DataFrame({"Classes": list(self.classes.values()),
                                 "avg_pred": self.avg_pred_train.cpu()})
        df_train.to_csv(path_or_buf=os.path.join(self.path_dest, "train_avg_preds.csv"),
                        sep=',', header=True, index=False, encoding='utf-8')

        if self.avg_pred_val is not None:
            df_val = pd.DataFrame({"Classes": list(self.classes.values()),
                                   "avg_pred": self.avg_pred_val.cpu()})
            df_val.to_csv(path_or_buf=os.path.join(self.path_dest, "val_avg_preds.csv"),
                          sep=',', header=True, index=False, encoding='utf-8')

    def get_top_freq_classes(self):
        n_top = 10
        top_idx = torch.argsort(-self.avg_pred_train.cpu())[:n_top]
        top_classes = np.array(list(self.classes.values()))[top_idx]
        print('Prior (train), first {} classes: {}'.format(n_top, top_classes))