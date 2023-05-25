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
            return 0.

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
            return 0.

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
            return 0.

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

class MFocalLoss(nn.Module):
  def __init__(self):
    super(MFocalLoss, self).__init__()

  def forward(self, pred, gt):
      pos_inds = gt.eq(1).float()
      neg_inds = gt.lt(1).float()
      #print(gt.shape)
      #print(pos_inds)
      #print(neg_inds)

      neg_weights = torch.pow(1 - gt, 4)

      loss = 0
      
      eps = 1e-12

      pos_loss = torch.log(pred + eps) * torch.pow(1 - pred, 2 - 2 * pred) * pos_inds
      neg_loss = torch.log(1 - pred + eps) * torch.pow(pred, 2) * neg_weights * neg_inds

      num_pos  = pos_inds.float().sum()
      pos_loss = pos_loss.sum()
      neg_loss = neg_loss.sum()

      if num_pos == 0:
        loss = loss - neg_loss
        return loss, 0, -neg_loss
      else:
        loss = loss - (pos_loss + neg_loss) / num_pos
        return loss, -pos_loss / num_pos, -neg_loss / num_pos

class SteepFocalLoss(nn.Module):
  def __init__(self):
    super(SteepFocalLoss, self).__init__()

  def forward(self, pred, gt):
      pos_inds = gt.eq(1).float()
      neg_inds = gt.lt(1).float()
      #print(gt.shape)
      #print(pos_inds)
      #print(neg_inds)

      neg_weights = torch.pow(1 - gt, 4)

      loss = 0
      
      eps = 1e-12

      pos_loss = torch.log(pred + eps) * (2 - torch.pow(2, pred)) * pos_inds
      neg_loss = torch.log(1 - pred + eps) * torch.pow(pred, 2) * neg_weights * neg_inds

      num_pos  = pos_inds.float().sum()
      pos_loss = pos_loss.sum()
      neg_loss = neg_loss.sum()

      if num_pos == 0:
        loss = loss - neg_loss
      else:
        loss = loss - (pos_loss + neg_loss) / num_pos
      return loss


class ClasswiseFocalLoss(nn.Module):
  def __init__(self):
    super(ClasswiseFocalLoss, self).__init__()

  def forward(self, pred, gt):
      
      loss = 0
      num_pos = 0
      
      # gamma_list = [2.8, 2.5, 2.0, 2.1, 2.3, 2.6, 2.8, 2.8, 2.4, 2.7, 2.0, 2.2, 2.4, 2.2, 2.3]  # efl8
      # gamma_list = [2.3, 2.9, 2.6, 3.0, 2.0, 2.1, 2.0, 2.5, 2.9, 2.4, 3.0, 2.9, 2.3, 2.6, 2.8]  # efl7
      # gamma_list = [2.0, 2.3, 2.8, 2.7, 2.5, 2.2, 2.0, 2.0, 2.4, 2.1, 2.8, 2.6, 2.4, 2.6, 2.5]  # efl6
      gamma_list = [2.7, 2.1, 2.4, 2.0, 3.0, 2.9, 3.0, 2.5, 2.1, 2.6, 2.0, 2.1, 2.7, 2.4, 2.2]  # efl3
      # gamma_list = [2.2, 1.6, 1.9, 1.5, 2.5, 2.4, 2.5, 2.0, 1.6, 2.1, 1.5, 1.6, 2.2, 1.9, 1.7]  # efl4
      # gamma_list = [3.4, 2.2, 2.8, 2.0, 4.0, 3.8, 4.0, 3.0, 2.2, 3.2, 2.0, 2.2, 3.4, 2.8, 2.4]  # efl5
      pred_sum = torch.sum(pred, dim=1).unsqueeze(1).repeat([1,15,1,1])
      pred_rest_sum = pred_sum - pred
      for i, gamma in enumerate(gamma_list):
        gt_cat = gt[:,i,:,:]
        pred_cat = pred[:,i,:,:]
        pred_rest_sum_cat = pred_rest_sum[:,i,:,:]
        pos_inds = gt_cat.eq(1).float()
        neg_inds = gt_cat.lt(1).float()
        neg_weights = torch.pow(1 - gt_cat, 4)
      
        eps = 1e-12

        pos_loss = torch.log(pred_cat + eps) * torch.pow(1 - pred_cat, gamma) * torch.pow(1 + pred_rest_sum_cat, 1) * pos_inds
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


class ClassConfusionLoss(nn.Module):
  def __init__(self):
    super(ClassConfusionLoss, self).__init__()

  def forward(self, pred, gt):
      pred = pred / (torch.sum(pred, dim=1) + 1e-12)
      pos_inds = gt.eq(1).float()
      (B, C, W, H) = gt.shape
      num_pos  = int(pos_inds.float().sum())
      train_bs = num_pos
      class_num = C
      if num_pos == 0:
          return 0
      else:
          index = torch.nonzero(pos_inds)
          # print(index.shape)
          for num in range(num_pos):
              if num == 0:
                  target_softmax_out_temp = pred[index[0,0], :, index[0,2], index[0,3]].unsqueeze(0)
              else:
                  target_softmax_out_temp = torch.cat((target_softmax_out_temp, pred[index[num,0], :, index[num,2], index[num,3]].unsqueeze(0)),0)
          # print(num_pos)
          # print(pred_softmax_out_temp.shape)
          epsilon = 1e-12
          target_entropy_weight = -target_softmax_out_temp * torch.log(target_softmax_out_temp + epsilon)
          target_entropy_weight = torch.sum(target_entropy_weight, dim=1)
          #target_entropy_weight = loss.Entropy(target_softmax_out_temp)#.detach()
          target_entropy_weight = 1 + torch.exp(-target_entropy_weight)
          target_entropy_weight = train_bs * target_entropy_weight / torch.sum(target_entropy_weight)
          # print(target_entropy_weight.sum())
          cov_matrix_t = target_softmax_out_temp.mul(target_entropy_weight.view(-1,1)).transpose(1,0).mm(target_softmax_out_temp)
          cov_matrix_t = cov_matrix_t / torch.sum(cov_matrix_t, dim=1)
          cc_loss = (torch.sum(cov_matrix_t) - torch.trace(cov_matrix_t)) / class_num
          return cc_loss

class EntropyLoss(nn.Module):
  def __init__(self):
    super(EntropyLoss, self).__init__()

  def forward(self, pred):
      eps = 1e-12
      entropy_map = torch.sum(-pred * torch.log(pred + eps), dim=1)
      loss = torch.mean(entropy_map)
      return loss

class Prototype3ContrastLoss(nn.Module):
  def __init__(self, tau=0.07):
      super(Prototype3ContrastLoss, self).__init__()
      self.tau = tau

  def forward(self, feat, gt):
      loss = 0
      
      feat = F.normalize(feat, dim=1, p=2)
      
      device = feat.device
      bs, c, h, w = feat.size()
      num_class = gt.size(1)
      # feat = F.normalize(feat, dim=1, p=2)
      pos_inds = gt.eq(1).float()
      # neg_inds = gt.sum(dim=1).eq(0).float()
      # pos_weight = pred * pos_inds
      # neg_weight = pred * neg_inds
      # num_pos  = int(pos_inds.float().sum())
      # index = torch.nonzero(pos_inds)
      protos = torch.zeros(num_class, c).to(device)
      # bg_protos = torch.zeros(num_class, c).to(device)
      fg_exist = torch.zeros(num_class).to(device)
      k0_feat = feat.flatten(2)  # [bs, 256, h*w]
      for i in range(num_class):
          class_pos_weight = pos_inds[:,i,:,:].flatten(1).unsqueeze(1)
          fg_proto = (k0_feat * class_pos_weight).sum(dim=-1).sum(dim=0)
          fg_proto_norm = F.normalize(fg_proto, dim=0, p=2)
          protos[i, :] = fg_proto_norm
          if int(pos_inds[:,i,:,:].sum()) > 0:
              fg_exist[i] = 1
      # class_neg_weight = neg_inds.flatten(1).unsqueeze(1)
      # bg_proto = (k0_feat * class_neg_weight).sum(dim=-1).sum(dim=0)
      # bg_proto_norm = F.normalize(bg_proto, dim=0, p=2)
      # protos[num_class, :] = bg_proto_norm
      for i in range(num_class):
          if fg_exist[i] == 1:
              sim = torch.mm(protos, protos[i, :].unsqueeze(1)) / self.tau
              sim = torch.exp(sim)
              sim_score = sim[i] / sim.sum()
              loss -= torch.log(sim_score)
      # sim = torch.mm(protos, protos[num_class, :].unsqueeze(1)) / self.tau
      # sim = torch.exp(sim)
      # sim_score = sim[i] / sim.sum()
      # loss -= torch.log(sim_score)
      loss /= fg_exist.sum()
      return loss

class Prototype2ContrastLoss(nn.Module):
  def __init__(self, tau=0.07):
      super(Prototype2ContrastLoss, self).__init__()
      self.tau = tau

  def forward(self, feat, pred, gt):
      loss = 0
      
      feat = F.normalize(feat, dim=1, p=2)
      
      device = feat.device
      bs, c, h, w = feat.size()
      num_class = gt.size(1)
      # feat = F.normalize(feat, dim=1, p=2)
      pos_inds = gt.eq(1).float()
      neg_inds = gt.sum(dim=1).eq(0).float()
      pos_weight = pred * pos_inds
      # neg_weight = pred * neg_inds
      # num_pos  = int(pos_inds.float().sum())
      # index = torch.nonzero(pos_inds)
      protos = torch.zeros(num_class + 1, c).to(device)
      # bg_protos = torch.zeros(num_class, c).to(device)
      fg_exist = torch.zeros(num_class).to(device)
      k0_feat = feat.flatten(2)  # [bs, 256, h*w]
      for i in range(num_class):
          class_pos_weight = pos_weight[:,i,:,:].flatten(1).unsqueeze(1)
          fg_proto = (k0_feat * class_pos_weight).sum(dim=-1).sum(dim=0)
          fg_proto_norm = F.normalize(fg_proto, dim=0, p=2)
          protos[i, :] = fg_proto_norm
          if int(pos_inds[:,i,:,:].sum()) > 0:
              fg_exist[i] = 1
      class_neg_weight = neg_inds.flatten(1).unsqueeze(1)
      bg_proto = (k0_feat * class_neg_weight).sum(dim=-1).sum(dim=0)
      bg_proto_norm = F.normalize(bg_proto, dim=0, p=2)
      protos[num_class, :] = bg_proto_norm
      for i in range(num_class):
          if fg_exist[i] == 1:
              sim = torch.mm(protos, protos[i, :].unsqueeze(1)) / self.tau
              sim = torch.exp(sim)
              sim_score = sim[i] / sim.sum()
              loss -= torch.log(sim_score)
      sim = torch.mm(protos, protos[num_class, :].unsqueeze(1)) / self.tau
      sim = torch.exp(sim)
      sim_score = sim[i] / sim.sum()
      loss -= torch.log(sim_score)
      loss /= (fg_exist.sum() + 1)
      return loss

class Prototype1ContrastLoss(nn.Module):
  def __init__(self, tau=0.07):
      super(Prototype1ContrastLoss, self).__init__()
      self.tau = tau

  def forward(self, feat, pred, gt):
      loss = 0
      
      feat = F.normalize(feat, dim=1, p=2)
      
      device = feat.device
      bs, c, h, w = feat.size()
      num_class = gt.size(1)
      # feat = F.normalize(feat, dim=1, p=2)
      pos_inds = gt.eq(1).float()
      neg_inds = gt.lt(1).float()
      pos_weight = pred * pos_inds
      neg_weight = pred * neg_inds
      num_pos  = int(pos_inds.float().sum())
      index = torch.nonzero(pos_inds)
      # fg_protos = torch.zeros(num_class, c).to(device)
      # bg_protos = torch.zeros(num_class, c).to(device)
      # fg_exist = torch.zeros(num_class).to(device)
      num = 0
      k0_feat = feat.flatten(2)  # [bs, 256, h*w]
      for i in range(num_class):
          class_pos_weight = pos_weight[:,i,:,:].flatten(1).unsqueeze(1)
          fg_proto = (k0_feat * class_pos_weight).sum(dim=-1).sum(dim=0)
          fg_proto_norm = F.normalize(fg_proto, dim=0, p=2)
          class_neg_weight = neg_weight[:,i,:,:].flatten(1).unsqueeze(1)
          bg_proto = (k0_feat * class_neg_weight).sum(dim=-1).sum(dim=0)
          bg_proto_norm = F.normalize(bg_proto, dim=0, p=2)
          # fg_protos[i, :] = fg_proto
          # bg_protos[i, :] = bg_proto
          if int(pos_inds[:,i,:,:].sum()) > 0 and int(neg_weight[:,i,:,:].sum()) > 0:
              # fg_exist[i] = 1
              num += 1
              sim = fg_proto_norm.dot(bg_proto_norm)
              # print(sim)
              loss -= torch.log(1 / (1 + torch.exp((sim - 1) / self.tau)))
      loss /= num
      return loss

class PrototypeContrastLoss(nn.Module):
  def __init__(self, tau=0.07):
      super(PrototypeContrastLoss, self).__init__()
      self.tau = tau

  def forward(self, feat, gt):
      '''
      loss = 0
      
      device = feat.device
      bs, c, h, w = feat.size()
      num_class = gt.size(1)
      feat = F.normalize(feat, dim=1, p=2)
      pos_inds = gt.eq(1).float()
      # num_pos  = int(pos_inds.float().sum())
      # index = torch.nonzero(pos_inds)
      k0 = torch.zeros(num_class, c).to(device)
      k0_feat = feat.flatten(2)  # [bs, 256, h*w]
      for i in range(num_class):
          weight = pos_inds[:,i,:,:].flatten(1).unsqueeze(1)
          k0_tmp = (k0_feat * weight).sum(dim=-1).sum(dim=0)
          k0[i, :] = F.normalize(k0_tmp, dim=0, p=2)
      for i in range(num_class):
          sim = torch.mm(k0, k0[i, :].unsqueeze(1)) / self.tau
          sim = torch.exp(sim)
          sim_score = sim[i] / sim.sum()
          loss -= torch.log(sim_score)
      loss /= num_class
      return loss
      '''
      loss = 0
      
      device = feat.device
      bs, c, h, w = feat.size()
      num_class = gt.size(1)
      # feat = F.normalize(feat, dim=1, p=2)
      pos_inds = gt.eq(1).float()
      num_pos  = int(pos_inds.float().sum())
      index = torch.nonzero(pos_inds)
      k0 = torch.zeros(num_class, c).to(device)
      k0_is = torch.zeros(num_class).to(device)
      for num in range(num_pos):
          feat_norm = F.normalize(feat[index[num, 0], :, index[num, 2], index[num, 3]], dim=0, p=2)
          k0[index[num, 1]] += feat_norm
          k0_is[index[num, 1]] = 1
      k0_norm = F.normalize(k0, dim=1, p=2)
      for i in range(num_class):
          if k0_is[i] == 1:
              sim = torch.mm(k0_norm, k0_norm[i, :].unsqueeze(1)) / self.tau
              # print(sim * self.tau)
              sim = torch.exp(sim)
              # print(sim)
              sim_score = sim[i] / sim.sum()
              loss -= torch.log(sim_score)
      loss /= k0_is.sum()
      return loss

class GroupContrastLoss(nn.Module):
  def __init__(self, tau=0.07):
      super(GroupContrastLoss, self).__init__()
      self.tau = tau

  def forward(self, feat, gt):
      loss = 0
      
      device = feat.device
      bs, c, h, w = feat.size()
      num_class = gt.size(1)
      # feat = F.normalize(feat, dim=1, p=2)
      pos_inds = gt.eq(1).float()
      num_pos  = int(pos_inds.float().sum())
      index = torch.nonzero(pos_inds)
      k0 = torch.zeros(num_class, c).to(device)
      for num in range(num_pos):
          feat_norm = F.normalize(feat[index[num, 0], :, index[num, 2], index[num, 3]], dim=0, p=2)
          k0[index[num, 1]] += feat_norm
      k0_norm = F.normalize(k0, dim=1, p=2)
      for num in range(num_pos):
          feat_norm = F.normalize(feat[index[num, 0], :, index[num, 2], index[num, 3]], dim=0, p=2)
          sim = torch.mm(k0_norm, feat_norm.unsqueeze(1)) / self.tau
          sim = torch.exp(sim)
          sim_norm = sim / sim.sum()
          # print(sim.shape)
          # print(sim_norm[index[num, 1]])
          loss -= torch.log(sim_norm[index[num, 1]])
      loss /= num_pos
      return loss

class ContrastLoss(nn.Module):
  def __init__(self, tau=0.07):
      super(ContrastLoss, self).__init__()
      self.tau = tau

  def forward(self, feat, gt):
      loss = 0
      
      device = feat.device
      bs, c, h, w = feat.size()
      num_class = gt.size(1)
      # feat = F.normalize(feat, dim=1, p=2)
      pos_inds = gt.eq(1).float()
      num_pos  = int(pos_inds.float().sum())
      index = torch.nonzero(pos_inds)
      k0 = torch.zeros(num_class, c).to(device)
      for num in range(num_pos):
          ins_feat = feat[index[num, 0], :, index[num, 2], index[num, 3]]
          k0[index[num, 1]] += ins_feat
      k0_norm = F.normalize(k0, dim=1, p=2)
      for num in range(num_pos):
          feat_norm = F.normalize(feat[index[num, 0], :, index[num, 2], index[num, 3]], dim=0, p=2)
          sim = torch.mm(k0_norm, feat_norm.unsqueeze(1)) / self.tau
          sim = torch.exp(sim)
          sim_norm = sim / sim.sum()
          # print(sim.shape)
          # print(sim_norm[index[num, 1]])
          loss -= torch.log(sim_norm[index[num, 1]])
      loss /= num_pos
      return loss

class EntropyFocalLoss(nn.Module):
  def __init__(self):
    super(EntropyFocalLoss, self).__init__()

  def forward(self, pred, gt):
      pos_inds = gt.eq(1).float()
      neg_inds = gt.lt(1).float()
      '''
      pos_inds_sum = torch.sum(pos_inds, dim=1).unsqueeze(1).repeat([1,15,1,1])
      neg_inds_pos = pos_inds_sum - pos_inds
      neg_inds_neg = 1 - pos_inds_sum
      '''
      #print(gt.shape)
      #print(pos_inds)
      #print(neg_inds)
      # pred_norm = nn.Softmax(dim=1)(pred)
      # print(pred_norm)
      eps = 1e-12
      entropy_map = torch.sum(-pred * torch.log(pred + eps), dim=1)
      # entropy_weight_pos = torch.exp(-entropy_map)
      # entropy_weight_neg = torch.exp(entropy_map)

      neg_weights = torch.pow(1 - gt, 4)

      loss = 0

      pos_loss = torch.log(pred + eps) * torch.pow(1 - pred, 2) * pos_inds
      neg_loss = torch.log(1 - pred + eps) * torch.pow(pred, 2) * neg_weights * neg_inds

      num_pos  = pos_inds.float().sum()
      pos_loss = pos_loss.sum()
      neg_loss = neg_loss.sum()

      avg_entropy = entropy_map.sum() / 152 / 152
      if num_pos == 0:
        loss = loss - neg_loss
      else:
        # avg_sample_entropy = entropy_map.sum() / num_pos
        # print(avg_sample_entropy)
        loss = loss - (pos_loss + neg_loss) / num_pos
      loss = loss + 0.3 * avg_entropy
      return loss

class EqualizedFocalLoss(nn.Module):
  def __init__(self):
    super(EqualizedFocalLoss, self).__init__()

  def forward(self, pred, gt):
      
      loss = 0
      num_pos = 0
      
      # gamma_list = [2.8, 2.5, 2.0, 2.1, 2.3, 2.6, 2.8, 2.8, 2.4, 2.7, 2.0, 2.2, 2.4, 2.2, 2.3]  # efl8
      # gamma_list = [2.3, 2.9, 2.6, 3.0, 2.0, 2.1, 2.0, 2.5, 2.9, 2.4, 3.0, 2.9, 2.3, 2.6, 2.8]  # efl7
      # gamma_list = [2.0, 2.3, 2.8, 2.7, 2.5, 2.2, 2.0, 2.0, 2.4, 2.1, 2.8, 2.6, 2.4, 2.6, 2.5]  # efl6
      gamma_list = [2.7, 2.1, 2.4, 2.0, 3.0, 2.9, 3.0, 2.5, 2.1, 2.6, 2.0, 2.1, 2.7, 2.4, 2.2]  # efl3
      # gamma_list = [2.2, 1.6, 1.9, 1.5, 2.5, 2.4, 2.5, 2.0, 1.6, 2.1, 1.5, 1.6, 2.2, 1.9, 1.7]  # efl4
      # gamma_list = [3.4, 2.2, 2.8, 2.0, 4.0, 3.8, 4.0, 3.0, 2.2, 3.2, 2.0, 2.2, 3.4, 2.8, 2.4]  # efl5
      # gamma_list = [2.0] * 15
      for i, gamma in enumerate(gamma_list):
        gt_cat = gt[:,i,:,:]
        pred_cat = pred[:,i,:,:]
        pos_inds = gt_cat.eq(1).float()
        neg_inds = gt_cat.lt(1).float()
        neg_weights = torch.pow(1 - gt_cat, 4)
      
        eps = 1e-12

        pos_loss = torch.log(pred_cat + eps) * torch.pow(1 - pred_cat, gamma) * pos_inds
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

class EqualizedMFocalLoss(nn.Module):
  def __init__(self):
    super(EqualizedMFocalLoss, self).__init__()

  def forward(self, pred, gt):
      
      loss = 0
      num_pos = 0
      
      # gamma_list = [2.8, 2.5, 2.0, 2.1, 2.3, 2.6, 2.8, 2.8, 2.4, 2.7, 2.0, 2.2, 2.4, 2.2, 2.3]  # efl8
      # gamma_list = [2.3, 2.9, 2.6, 3.0, 2.0, 2.1, 2.0, 2.5, 2.9, 2.4, 3.0, 2.9, 2.3, 2.6, 2.8]  # efl7
      # gamma_list = [2.0, 2.3, 2.8, 2.7, 2.5, 2.2, 2.0, 2.0, 2.4, 2.1, 2.8, 2.6, 2.4, 2.6, 2.5]  # efl6
      gamma_list = [2.7, 2.1, 2.4, 2.0, 3.0, 2.9, 3.0, 2.5, 2.1, 2.6, 2.0, 2.1, 2.7, 2.4, 2.2]  # efl3
      # gamma_list = [2.7, 2.1, 2.4, 2.0, 3.0, 2.9, 2.9, 2.5, 2.1, 2.6, 2.0, 2.1, 2.7, 2.4, 2.2, 2.0]  # efl3 dota15
      # gamma_list = [2.2, 1.6, 1.9, 1.5, 2.5, 2.4, 2.5, 2.0, 1.6, 2.1, 1.5, 1.6, 2.2, 1.9, 1.7]  # efl4
      # gamma_list = [3.4, 2.2, 2.8, 2.0, 4.0, 3.8, 4.0, 3.0, 2.2, 3.2, 2.0, 2.2, 3.4, 2.8, 2.4]  # efl5
      # gamma_list = [2.0] * 15
      for i, gamma in enumerate(gamma_list):
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

class EqualizedFocalLoss1(nn.Module):
  def __init__(self):
    super(EqualizedFocalLoss1, self).__init__()
  
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

  def forward(self, pred, gt, output, mask, ind, target, inde):
      
      loss = 0
      num_pos = 0
      
      # gamma_list = [2.8, 2.5, 2.0, 2.1, 2.3, 2.6, 2.8, 2.8, 2.4, 2.7, 2.0, 2.2, 2.4, 2.2, 2.3]  # efl8
      # gamma_list = [2.3, 2.9, 2.6, 3.0, 2.0, 2.1, 2.0, 2.5, 2.9, 2.4, 3.0, 2.9, 2.3, 2.6, 2.8]  # efl7
      # gamma_list = [2.0, 2.3, 2.8, 2.7, 2.5, 2.2, 2.0, 2.0, 2.4, 2.1, 2.8, 2.6, 2.4, 2.6, 2.5]  # efl6
      gamma_list = [2.7, 2.1, 2.4, 2.0, 3.0, 2.9, 3.0, 2.5, 2.1, 2.6, 2.0, 2.1, 2.7, 2.4, 2.2]  # efl3
      # gamma_list = [2.2, 1.6, 1.9, 1.5, 2.5, 2.4, 2.5, 2.0, 1.6, 2.1, 1.5, 1.6, 2.2, 1.9, 1.7]  # efl4
      # gamma_list = [3.4, 2.2, 2.8, 2.0, 4.0, 3.8, 4.0, 3.0, 2.2, 3.2, 2.0, 2.2, 3.4, 2.8, 2.4]  # efl5
      pred_uni = pred
      pre = self._tranpose_and_gather_feat(output, ind)  # torch.Size([1, 500, 2])
      if mask.sum():
            num = mask.sum()
            index = torch.nonzero(mask)
            for i in range(num):
                mask1 = torch.zeros_like(mask)
                mask1[index[i][0],index[i][1]] = 1
                # print(inde.shape)
                # print(mask1.shape)
                # p = hm[index[i][0], inde[index[i][0],index[i][1],0], inde[index[i][0],index[i][1],1], inde[index[i][0],index[i][1],2]]
                # print(p)
                mask1 = mask1.unsqueeze(2).expand_as(pre).bool()
                # print(pred.masked_select(mask1))
                # print(target.masked_select(mask1))
                loss = F.smooth_l1_loss(pre.masked_select(mask1),
                                    target.masked_select(mask1),
                                    reduction='mean')
                # print(pred_uni[index[i][0], inde[index[i][0],index[i][1],0], inde[index[i][0],index[i][1],1], inde[index[i][0],index[i][1],2]])
                pred_uni[index[i][0], inde[index[i][0],index[i][1],0], inde[index[i][0],index[i][1],1], inde[index[i][0],index[i][1],2]] *= (math.atan(loss)*2/math.pi)
                # print(pred_uni[index[i][0], inde[index[i][0],index[i][1],0], inde[index[i][0],index[i][1],1], inde[index[i][0],index[i][1],2]])
      for i, gamma in enumerate(gamma_list):
        gt_cat = gt[:,i,:,:]
        pred_cat = pred[:,i,:,:]
        pred_cat_uni = pred_uni[:,i,:,:]
        pos_inds = gt_cat.eq(1).float()
        neg_inds = gt_cat.lt(1).float()
        neg_weights = torch.pow(1 - gt_cat, 4)
      
        eps = 1e-12

        pos_loss = torch.log(pred_cat + eps) * torch.pow(1 - pred_cat_uni, gamma) * pos_inds
        neg_loss = torch.log(1 - pred_cat + eps) * torch.pow(pred_cat_uni, gamma) * neg_weights * neg_inds

        num_pos  = pos_inds.float().sum() + num_pos
        pos_loss = pos_loss.sum()
        neg_loss = neg_loss.sum()
        loss = loss - (pos_loss + neg_loss) * gamma / 2

      if num_pos == 0:
        loss = loss
      else:
        loss = loss / num_pos

      return loss

class EqualizedPolyLoss(nn.Module):
  def __init__(self):
    super(EqualizedPolyLoss, self).__init__()

  def forward(self, pred, gt):
      
      loss = 0
      num_pos = 0
      
      # gamma_list = [2.8, 2.5, 2.0, 2.1, 2.3, 2.6, 2.8, 2.8, 2.4, 2.7, 2.0, 2.2, 2.4, 2.2, 2.3]  # efl8
      # gamma_list = [2.3, 2.9, 2.6, 3.0, 2.0, 2.1, 2.0, 2.5, 2.9, 2.4, 3.0, 2.9, 2.3, 2.6, 2.8]  # efl7
      # gamma_list = [2.0, 2.3, 2.8, 2.7, 2.5, 2.2, 2.0, 2.0, 2.4, 2.1, 2.8, 2.6, 2.4, 2.6, 2.5]  # efl6
      gamma_list = [2.7, 2.1, 2.4, 2.0, 3.0, 2.9, 3.0, 2.5, 2.1, 2.6, 2.0, 2.1, 2.7, 2.4, 2.2]  # efl3
      # gamma_list = [2.2, 1.6, 1.9, 1.5, 2.5, 2.4, 2.5, 2.0, 1.6, 2.1, 1.5, 1.6, 2.2, 1.9, 1.7]  # efl4
      # gamma_list = [3.4, 2.2, 2.8, 2.0, 4.0, 3.8, 4.0, 3.0, 2.2, 3.2, 2.0, 2.2, 3.4, 2.8, 2.4]  # efl5
      for i, gamma in enumerate(gamma_list):
        gt_cat = gt[:,i,:,:]
        pred_cat = pred[:,i,:,:]
        pos_inds = gt_cat.eq(1).float()
        neg_inds = gt_cat.lt(1).float()
        neg_weights = torch.pow(1 - gt_cat, 4)
      
        eps = 1e-12

        pos_loss = (torch.log(pred_cat + eps) * torch.pow(1 - pred_cat, gamma) + torch.pow(1 - pred_cat, gamma + 1) - torch.pow(1 - pred_cat, gamma + 2)) * pos_inds
        #print(((torch.log(pred_cat + eps) * torch.pow(1 - pred_cat, gamma) + torch.pow(1 - pred_cat, gamma + 1) - torch.pow(1 - pred_cat, gamma + 2)) * pos_inds).sum())
        #print((torch.log(pred_cat + eps) * torch.pow(1 - pred_cat, gamma) * pos_inds).sum())
        neg_loss = (torch.log(1 - pred_cat + eps) * torch.pow(pred_cat, gamma) + torch.pow( pred_cat, gamma + 1) - torch.pow(pred_cat, gamma + 2)) * neg_weights * neg_inds

        num_pos  = pos_inds.float().sum() + num_pos
        pos_loss = pos_loss.sum()
        neg_loss = neg_loss.sum()
        loss = loss - (pos_loss + neg_loss) * gamma / 2

      if num_pos == 0:
        loss = loss
      else:
        loss = loss / num_pos

      return loss

class EqualizedFLoss(nn.Module):
  def __init__(self):
    super(EqualizedFLoss, self).__init__()

  def forward(self, pred, gt):
      
      loss = 0
      num_pos = 0
      
      # gamma_list = [2.8, 2.5, 2.0, 2.1, 2.3, 2.6, 2.8, 2.8, 2.4, 2.7, 2.0, 2.2, 2.4, 2.2, 2.3]  # efl8
      # gamma_list = [2.3, 2.9, 2.6, 3.0, 2.0, 2.1, 2.0, 2.5, 2.9, 2.4, 3.0, 2.9, 2.3, 2.6, 2.8]  # efl7
      # gamma_list = [2.0, 2.3, 2.8, 2.7, 2.5, 2.2, 2.0, 2.0, 2.4, 2.1, 2.8, 2.6, 2.4, 2.6, 2.5]  # efl6
      gamma_list = [2.7, 2.1, 2.4, 2.0, 3.0, 2.9, 3.0, 2.5, 2.1, 2.6, 2.0, 2.1, 2.7, 2.4, 2.2]  # efl3
      # gamma_list = [2.2, 1.6, 1.9, 1.5, 2.5, 2.4, 2.5, 2.0, 1.6, 2.1, 1.5, 1.6, 2.2, 1.9, 1.7]  # efl4
      # gamma_list = [3.4, 2.2, 2.8, 2.0, 4.0, 3.8, 4.0, 3.0, 2.2, 3.2, 2.0, 2.2, 3.4, 2.8, 2.4]  # efl5
      for i, gamma in enumerate(gamma_list):
        gt_cat = gt[:,i,:,:]
        pred_cat = pred[:,i,:,:]
        pos_inds = gt_cat.eq(1).float()
        neg_inds = gt_cat.lt(1).float()
        neg_weights = torch.pow(1 - gt_cat, 4)
      
        eps = 1e-12

        pos_loss = (torch.log(pred_cat + eps) * torch.pow(1 - pred_cat, gamma) - torch.pow(1 - pred_cat, gamma + 1) + torch.pow(1 - pred_cat, gamma + 2)) * pos_inds
        #print(((torch.log(pred_cat + eps) * torch.pow(1 - pred_cat, gamma) + torch.pow(1 - pred_cat, gamma + 1) - torch.pow(1 - pred_cat, gamma + 2)) * pos_inds).sum())
        #print((torch.log(pred_cat + eps) * torch.pow(1 - pred_cat, gamma) * pos_inds).sum())
        neg_loss = (torch.log(1 - pred_cat + eps) * torch.pow(pred_cat, gamma) - torch.pow( pred_cat, gamma + 1) + torch.pow(pred_cat, gamma + 2)) * neg_weights * neg_inds

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

    def forward(self, pr_decs, feat, soft_label, gt_batch, epoch):
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
