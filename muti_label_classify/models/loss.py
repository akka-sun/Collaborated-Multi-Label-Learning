import torch
import torch.nn as nn
import numpy as np

class BCELoss(nn.Module):
    def __init__(self):
        super(BCELoss, self).__init__()
    def forward(self, x, y):
        """"
        Parameters
        ----------
        x: input logits
        y: targets (multi-label binarized vector)
        """
        eps = 1e-8
        x_sigmoid = torch.sigmoid(x)
        x_sigmoid = x_sigmoid.squeeze()
        bce_loss = -(torch.log(x_sigmoid + eps) * y + torch.log(1 - x_sigmoid + eps) * (1 - y))
        loss = torch.sum(bce_loss,dim=1)
        loss = torch.mean(loss)

        return loss

class MyLoss(nn.Module):
    def __init__(self):
        super(MyLoss, self).__init__()

    def forward(self, x, y,weight_01,weight_00,org_idx):
        # Calculating Probabilities
        eps = 1e-8
        r = 0.1
        weight_01 = r*weight_01+(1-r)*y
        weight_00 = 1 - weight_01
        org_idx = org_idx.cpu().numpy()
        rows, cols = np.where(org_idx == 0)
        rows = torch.from_numpy(rows).cuda(non_blocking=True)
        cols = torch.from_numpy(cols).cuda(non_blocking=True)
        x_sigmoid = torch.sigmoid(x)
        x_sigmoid = x_sigmoid.squeeze()
        bce_loss = -(torch.log(x_sigmoid+eps)*y+torch.log(1-x_sigmoid+eps)*(1-y))
        total_loss = bce_loss.clone()
        total_loss[rows,cols] = bce_loss[rows,cols] * weight_00[rows,cols] - torch.log(x_sigmoid[rows,cols])*weight_01[rows,cols]

        loss = torch.sum(total_loss, dim=1)
        last_loss = torch.mean(loss)
        return last_loss


import torch
import torch.nn as nn

nINF = -100


class TwoWayLoss(nn.Module):
    def __init__(self, Tp=4., Tn=1.):
        super(TwoWayLoss, self).__init__()
        self.Tp = Tp
        self.Tn = Tn

    def forward(self, x, y):
        x = x.squeeze()
        class_mask = (y > 0).any(dim=0)
        sample_mask = (y > 0).any(dim=1)

        # Calculate hard positive/negative logits
        pmask = y.masked_fill(y <= 0, nINF).masked_fill(y > 0, float(0.0))
        plogit_class = torch.logsumexp(-x / self.Tp + pmask, dim=0).mul(self.Tp)[class_mask]
        plogit_sample = torch.logsumexp(-x / self.Tp + pmask, dim=1).mul(self.Tp)[sample_mask]

        nmask = y.masked_fill(y != 0, nINF).masked_fill(y == 0, float(0.0))
        nlogit_class = torch.logsumexp(x / self.Tn + nmask, dim=0).mul(self.Tn)[class_mask]
        nlogit_sample = torch.logsumexp(x / self.Tn + nmask, dim=1).mul(self.Tn)[sample_mask]

        return torch.nn.functional.softplus(nlogit_class + plogit_class).mean() + \
            torch.nn.functional.softplus(nlogit_sample + plogit_sample).mean()

class AsymmetricLoss(nn.Module):
    def __init__(self, gamma_neg=4, gamma_pos=1, clip=0.05, eps=1e-8, disable_torch_grad_focal_loss=True):
        super(AsymmetricLoss, self).__init__()

        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.disable_torch_grad_focal_loss = disable_torch_grad_focal_loss
        self.eps = eps

    def forward(self, x, y):
        """"
        Parameters
        ----------
        x: input logits
        y: targets (multi-label binarized vector)
        """

        # Calculating Probabilities
        x_sigmoid = torch.sigmoid(x)
        xs_pos = x_sigmoid
        xs_neg = 1 - x_sigmoid

        # Asymmetric Clipping
        if self.clip is not None and self.clip > 0:
            xs_neg = (xs_neg + self.clip).clamp(max=1)

        # Basic CE calculation
        los_pos = y * torch.log(xs_pos.clamp(min=self.eps))
        los_neg = (1 - y) * torch.log(xs_neg.clamp(min=self.eps))
        loss = los_pos + los_neg

        # Asymmetric Focusing
        if self.gamma_neg > 0 or self.gamma_pos > 0:
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(False)
            pt0 = xs_pos * y
            pt1 = xs_neg * (1 - y)  # pt = p if t > 0 else 1-p
            pt = pt0 + pt1
            one_sided_gamma = self.gamma_pos * y + self.gamma_neg * (1 - y)
            one_sided_w = torch.pow(1 - pt, one_sided_gamma)
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(True)
            loss *= one_sided_w

        return -loss.sum()


def get_criterion(args):
    if args.loss == 'TwoWayLoss':
        return TwoWayLoss(Tp=args.loss_Tp, Tn=args.loss_Tn)
    else:
        raise ValueError(f"Not supported loss {args.loss}")

if __name__ == '__main__':
    loss = TwoWayLoss()
    mean = 0.0  # 均值
    std = 1.0  # 标准差
    x = torch.normal(mean=mean, std=std, size=(24,1,80)).to('cuda')
    y = torch.normal(mean=mean, std=std, size=(24,80)).to('cuda')
    weight_00 = torch.normal(mean=mean, std=std, size=(24, 80)).to('cuda')
    weight_01 = torch.normal(mean=mean, std=std, size=(24, 80)).to('cuda')
    ord_index = torch.normal(mean=mean, std=std, size=(24, 80)).to('cuda')
    ord_index[0,0] = 0
    ord_index[0,1] = 0
    ord_index[0,2] = 0
    ord_index[1, 0] = 0
    print(loss(x,y))

