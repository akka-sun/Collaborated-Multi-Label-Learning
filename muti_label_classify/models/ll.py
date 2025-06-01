from typing_extensions import final
import torch
from torch._C import ThroughputBenchmark
import torch.nn.functional as F
import math

'''
loss functions
'''


def loss_an(logits, observed_labels):
    assert torch.min(observed_labels) >= 0
    # compute loss:
    loss_matrix = F.binary_cross_entropy_with_logits(logits, observed_labels, reduction='none')
    corrected_loss_matrix = F.binary_cross_entropy_with_logits(logits, torch.logical_not(observed_labels).float(),
                                                               reduction='none')
    return loss_matrix, corrected_loss_matrix


'''
top-level wrapper
'''


def compute_batch_loss(preds, label_vec, cr):  # "preds" are actually logits (not sigmoid activated !)

    assert preds.dim() == 2

    batch_size = int(preds.size(0))
    num_classes = int(preds.size(1))

    unobserved_mask = (label_vec == 0)

    # compute loss for each image and class:
    loss_matrix, corrected_loss_matrix = loss_an(preds, label_vec.clip(0))

    correction_idx = None

    if cr == 1:  # if epoch is 1, do not modify losses
        final_loss_matrix = loss_matrix
    else:
        k = math.ceil(batch_size * num_classes * (1 - cr))

        unobserved_loss = unobserved_mask.bool() * loss_matrix
        topk = torch.topk(unobserved_loss.flatten(), k)
        topk_lossvalue = topk.values[-1]
        correction_idx = torch.where(unobserved_loss > topk_lossvalue)
        zero_loss_matrix = torch.zeros_like(loss_matrix)
        final_loss_matrix = torch.where(unobserved_loss < topk_lossvalue, loss_matrix, zero_loss_matrix)

    main_loss = final_loss_matrix.mean()

    return main_loss