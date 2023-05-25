import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class BCELoss(nn.Module):
    def __init__(self):
        super(BCELoss, self).__init__()

    def _gather_feat(self, feat, ind, mask=None):
        dim = feat.size(2)
        ind = ind.unsqueeze(2).expand(ind.size(0), ind.size(1), dim)
        feat = feat.gather(1, ind)
        if mask is not None:
            mask = mask.unsqueeze(2).expand_as(feat)
            feat = feat[mask]
            feat = feat.view(-1, dim)
        return feat

    def _tranpose_and_gather_feat(self, feat, ind):
        feat = feat.permute(0, 2, 3, 1).contiguous()
        feat = feat.view(feat.size(0), -1, feat.size(3))
        feat = self._gather_feat(feat, ind)
        return feat

    def forward(self, output, mask, ind, target):
        # torch.Size([1, 1, 152, 152])
        # torch.Size([1, 500])
        # torch.Size([1, 500])
        # torch.Size([1, 500, 1])
        pred = self._tranpose_and_gather_feat(output, ind)  # torch.Size([1, 500, 1])
        if mask.sum():
            mask = mask.unsqueeze(2).expand_as(pred).bool()
            loss = F.binary_cross_entropy(pred.masked_select(mask),
                                          target.masked_select(mask),
                                          reduction='mean')
            return loss
        else:
            return torch.zeros(1).to(device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))

class OffSmoothL1Loss(nn.Module):
    def __init__(self):
        super(OffSmoothL1Loss, self).__init__()

    def _gather_feat(self, feat, ind, mask=None):
        dim = feat.size(2)
        ind = ind.unsqueeze(2).expand(ind.size(0), ind.size(1), dim)
        feat = feat.gather(1, ind)
        if mask is not None:
            mask = mask.unsqueeze(2).expand_as(feat)
            feat = feat[mask]
            feat = feat.view(-1, dim)
        return feat

    def _tranpose_and_gather_feat(self, feat, ind):
        feat = feat.permute(0, 2, 3, 1).contiguous()
        feat = feat.view(feat.size(0), -1, feat.size(3))
        feat = self._gather_feat(feat, ind)
        return feat

    def forward(self, output, mask, ind, target):
        # torch.Size([1, 2, 152, 152])
        # torch.Size([1, 500])
        # torch.Size([1, 500])
        # torch.Size([1, 500, 2])
        pred = self._tranpose_and_gather_feat(output, ind)  # torch.Size([1, 500, 2])
        if mask.sum():
            mask = mask.unsqueeze(2).expand_as(pred).bool()
            loss = F.smooth_l1_loss(pred.masked_select(mask),
                                    target.masked_select(mask),
                                    reduction='mean')
            return loss
        else:
            return torch.zeros(1).to(device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))

class SmoothL1Loss(nn.Module):
    def __init__(self):
        super(SmoothL1Loss, self).__init__()

    def forward(self, label, am):
        loss = F.smooth_l1_loss(label, am, reduction='mean')
        return loss

class OffSmoothL1LossPlus(nn.Module):
    def __init__(self):
        super(OffSmoothL1LossPlus, self).__init__()

    def _gather_feat(self, feat, ind, mask=None):
        dim = feat.size(2)
        ind = ind.unsqueeze(2).expand(ind.size(0), ind.size(1), dim)
        feat = feat.gather(1, ind)
        if mask is not None:
            mask = mask.unsqueeze(2).expand_as(feat)
            feat = feat[mask]
            feat = feat.view(-1, dim)
        return feat

    def _tranpose_and_gather_feat(self, feat, ind):
        feat = feat.permute(0, 2, 3, 1).contiguous()
        feat = feat.view(feat.size(0), -1, feat.size(3))
        feat = self._gather_feat(feat, ind)
        return feat

    def forward(self, output, mask, ind, target, inde, hm):
        # torch.Size([1, 2, 152, 152])
        # torch.Size([1, 500])
        # torch.Size([1, 500])
        # torch.Size([1, 500, 2])
        loss = 0
        pred = self._tranpose_and_gather_feat(output, ind)  # torch.Size([1, 500, 2])
        if mask.sum():
            num = mask.sum()
            index = torch.nonzero(mask)
            for i in range(num):
                mask1 = torch.zeros_like(mask)
                mask1[index[i][0],index[i][1]] = 1
                # print(inde.shape)
                # print(mask1.shape)
                p = hm[index[i][0], inde[index[i][0],index[i][1],0], inde[index[i][0],index[i][1],1], inde[index[i][0],index[i][1],2]]
                # print(p)
                mask1 = mask1.unsqueeze(2).expand_as(pred).bool()
                # print(pred.masked_select(mask1))
                # print(target.masked_select(mask1))
                loss += F.smooth_l1_loss(pred.masked_select(mask1),
                                    target.masked_select(mask1),
                                    reduction='mean') * torch.pow(1 + p, 2)
            loss /= num
            # print(pred.masked_select(mask).shape)
            return loss
        else:
            return torch.zeros(1).to(device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))

class FocalLoss(nn.Module):
  def __init__(self):
    super(FocalLoss, self).__init__()

  def forward(self, pred, gt):
      pos_inds = gt.eq(1).float()
      neg_inds = gt.lt(1).float()
      #print(gt.shape)
      #print(pos_inds)
      #print(neg_inds)

      neg_weights = torch.pow(1 - gt, 4)

      loss = 0
      
      eps = 1e-12

      pos_loss = torch.log(pred + eps) * torch.pow(1 - pred, 2) * pos_inds
      neg_loss = torch.log(1 - pred + eps) * torch.pow(pred, 2) * neg_weights * neg_inds

      num_pos  = pos_inds.float().sum()
      pos_loss = pos_loss.sum()
      neg_loss = neg_loss.sum()

      if num_pos == 0:
        loss = loss - neg_loss
      else:
        loss = loss - (pos_loss + neg_loss) / num_pos
      return loss



class EqualizedMFocalLoss(nn.Module):
  def __init__(self):
    super(EqualizedMFocalLoss, self).__init__()

  def forward(self, pred, gt):
      
      loss = 0
      num_pos = 0
      
      _, c, _, _ = gt.shape
      
      gamma = 2
      for i in range(c):
        gt_cat = gt[:,i,:,:]
        pred_cat = pred[:,i,:,:]
        pos_inds = gt_cat.eq(1).float()
        neg_inds = gt_cat.lt(1).float()
        neg_weights = torch.pow(1 - gt_cat, 4)
      
        eps = 1e-12

        pos_loss = torch.log(pred_cat + eps) * torch.pow(1 - pred_cat, gamma - gamma * pred_cat) * pos_inds
        neg_loss = torch.log(1 - pred_cat + eps) * torch.pow(pred_cat, gamma) * neg_weights * neg_inds

        num_pos  = pos_inds.float().sum() + num_pos
        pos_loss = pos_loss.sum()
        neg_loss = neg_loss.sum()
        loss = loss - (pos_loss + neg_loss) * gamma / 2

      if num_pos == 0:
        loss = loss
      else:
        loss = loss / num_pos

      return loss



def isnan(x):
    return x != x

  
class LossAll(torch.nn.Module):
    def __init__(self):
        super(LossAll, self).__init__()
        # self.L_cl = Prototype3ContrastLoss()
        # self.L_en = EntropyLoss()
        self.L_hm = EqualizedMFocalLoss()
        self.L_wh =  OffSmoothL1Loss()
        self.L_off = OffSmoothL1Loss()
        self.L_cls_theta = BCELoss()
        self.L_attention = SmoothL1Loss()

    def forward(self, pr_decs, soft_label, gt_batch):
        # cc_loss  = self.L_cc(pr_decs['hm'], gt_batch['hm'])
        # cl_loss = self.L_cl(feat, gt_batch['hm'])
        # en_loss = self.L_en(pr_decs['hm'])
        hm_loss  = self.L_hm(pr_decs['hm'], gt_batch['hm'])
        # wh_loss  = self.L_wh(pr_decs['wh'], gt_batch['reg_mask'], gt_batch['ind'], gt_batch['wh'], gt_batch['index'], pr_decs['hm'])
        wh_loss  = self.L_wh(pr_decs['wh'], gt_batch['reg_mask'], gt_batch['ind'], gt_batch['wh'])
        off_loss = self.L_off(pr_decs['reg'], gt_batch['reg_mask'], gt_batch['ind'], gt_batch['reg'])
        ## add
        cls_theta_loss = self.L_cls_theta(pr_decs['cls_theta'], gt_batch['reg_mask'], gt_batch['ind'], gt_batch['cls_theta'])
        attention_loss = self.L_attention(soft_label, gt_batch['am'])

        if isnan(hm_loss) or isnan(wh_loss) or isnan(off_loss):
            print('hm loss is {}'.format(hm_loss))
            print('wh loss is {}'.format(wh_loss))
            print('off loss is {}'.format(off_loss))

        # print(hm_loss)
        # print(wh_loss)
        # print(off_loss)
        # print(cls_theta_loss)
        # print('-----------------')

        loss =  hm_loss + wh_loss + off_loss + cls_theta_loss + attention_loss
        return loss, hm_loss, wh_loss, off_loss, cls_theta_loss, attention_loss
