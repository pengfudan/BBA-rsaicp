import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import torch
from .model_parts import CombinationModule
from . import resnet

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class ChannelGate(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max']):
        super(ChannelGate, self).__init__()
        self.gate_channels = gate_channels
        self.mlp = nn.Sequential(
            Flatten(),
            nn.Linear(gate_channels, gate_channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(gate_channels // reduction_ratio, gate_channels)
            )
        self.pool_types = pool_types
    def forward(self, x):
        channel_att_sum = None
        for pool_type in self.pool_types:
            if pool_type=='avg':
                avg_pool = F.avg_pool2d( x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp( avg_pool )
            elif pool_type=='max':
                max_pool = F.max_pool2d( x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp( max_pool )
            elif pool_type=='lp':
                lp_pool = F.lp_pool2d( x, 2, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp( lp_pool )
            elif pool_type=='lse':
                # LSE pool only
                lse_pool = logsumexp_2d(x)
                channel_att_raw = self.mlp( lse_pool )

            if channel_att_sum is None:
                channel_att_sum = channel_att_raw
            else:
                channel_att_sum = channel_att_sum + channel_att_raw

        scale = torch.sigmoid( channel_att_sum ).unsqueeze(2).unsqueeze(3).expand_as(x)
        return x * scale

class AC_Block(nn.Module):
    def __init__(self,
                 in_channels: int = 256,
                 out_channels: int = 256):
        super(AC_Block, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=(3,3), padding=(1,1), bias=True),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=(3,1), padding=(1,0), bias=True),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=(1,3), padding=(0,1), bias=True),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x3 = self.conv3(x)
        out = x1 + x2 + x3
        return out

class DB_Block(nn.Module):
    def __init__(self,
                 in_channels: int = 512,
                 mid_channels: int = 128,
                 dilation: int = 1):
        super(DB_Block, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=1, padding=0),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(mid_channels, mid_channels,
                      kernel_size=3, padding=dilation, dilation=dilation),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(mid_channels, in_channels, kernel_size=1, padding=0),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = out + identity
        return out

class CTRBOX(nn.Module):
    def __init__(self, heads, pretrained, down_ratio, final_kernel, head_conv, model="resnet18"):
        super(CTRBOX, self).__init__()
        assert down_ratio in [2, 4, 8, 16]
        self.l1 = int(np.log2(down_ratio))
        if model == "wide_resnet101_2":
            self.base_network = resnet.wide_resnet101_2(pretrained=pretrained)
            channels = [3, 64, 256, 512, 1024, 2048]
        elif model == "wide_resnet50_2":
            self.base_network = resnet.wide_resnet50_2(pretrained=pretrained)
            channels = [3, 64, 256, 512, 1024, 2048]
        elif model == "resnext101_32x8d":
            self.base_network = resnet.resnext101_32x8d(pretrained=pretrained)
            channels = [3, 64, 256, 512, 1024, 2048]
        elif model == "resnext50_32x4d":
            self.base_network = resnet.resnext50_32x4d(pretrained=pretrained)
            channels = [3, 64, 256, 512, 1024, 2048]
        elif model == "resnet101":
            self.base_network = resnet.resnet101(pretrained=pretrained)
            channels = [3, 64, 256, 512, 1024, 2048]
        elif model == "resnet50":
            self.base_network = resnet.resnet50(pretrained=pretrained)
            channels = [3, 64, 256, 512, 1024, 2048]
        elif model == "resnet34":
            self.base_network = resnet.resnet34(pretrained=pretrained)
            channels = [3, 64, 64, 128, 256, 512]
        elif model == "resnet18":
            self.base_network = resnet.resnet18(pretrained=pretrained)
            channels = [3, 64, 64, 128, 256, 512]
        else:
            raise ValueError("[Error]: "
                             "Parameter --> args.model_select must in [resnet101, resnet50, resnet34, resnet18]!")
        if down_ratio == 2:
            self.dec_c1 = CombinationModule(channels[2], channels[self.l1], batch_norm=True)
            self.dec_c2 = CombinationModule(channels[3], channels[2], batch_norm=True)
            self.dec_c3 = CombinationModule(channels[4], channels[3], batch_norm=True)
            self.dec_c4 = CombinationModule(channels[5], channels[4], batch_norm=True)
        elif down_ratio == 4:
            self.dec_c1 = None
            self.dec_c2 = CombinationModule(channels[3], channels[self.l1], batch_norm=True)
            self.dec_c3 = CombinationModule(channels[4], channels[3], batch_norm=True)
            self.dec_c4 = CombinationModule(channels[5], channels[4], batch_norm=True)
        elif down_ratio == 8:
            self.dec_c1 = None
            self.dec_c2 = None
            self.dec_c3 = CombinationModule(channels[4], channels[self.l1], batch_norm=True)
            self.dec_c4 = CombinationModule(channels[5], channels[4], batch_norm=True)
        elif down_ratio == 16:
            self.dec_c1 = None
            self.dec_c2 = None
            self.dec_c3 = None
            self.dec_c4 = CombinationModule(channels[5], channels[self.l1], batch_norm=True)
        self.heads = heads
        self.ChannelGate = ChannelGate(head_conv)
        self.DB_Blocks = nn.Sequential(nn.Conv2d(head_conv, head_conv, kernel_size=1, stride=1),
                                       nn.BatchNorm2d(head_conv),
                                       nn.Conv2d(head_conv, head_conv, kernel_size=3, padding=1),
                                       nn.BatchNorm2d(head_conv),
                                       DB_Block(head_conv,head_conv,dilation=2),
                                       DB_Block(head_conv,head_conv,dilation=2),
                                       DB_Block(head_conv,head_conv,dilation=2))
        self.final_conv = nn.Sequential(nn.Conv2d(head_conv, 1, kernel_size=1, padding=0, bias=True),
                                       nn.BatchNorm2d(1),
                                       nn.ReLU(inplace=True))

        for head in self.heads:
            classes = self.heads[head]
            if head == 'wh':
                fc = nn.Sequential(nn.Conv2d(channels[self.l1], head_conv, kernel_size=3, padding=1, bias=True),
                                   nn.BatchNorm2d(head_conv),   # BN not used in the paper, but would help stable training
                                   nn.ReLU(inplace=True),
                                   nn.Conv2d(head_conv, classes, kernel_size=3, padding=1, bias=True))
            else:
                fc = nn.Sequential(nn.Conv2d(channels[self.l1], head_conv, kernel_size=3, padding=1, bias=True),
                                   nn.BatchNorm2d(head_conv),   # BN not used in the paper, but would help stable training
                                   nn.ReLU(inplace=True),
                                   nn.Conv2d(head_conv, classes, kernel_size=final_kernel, stride=1, padding=final_kernel // 2, bias=True))
            if 'hm' in head:
                fc[-1].bias.data.fill_(-2.19)
            else:
                self.fill_fc_weights(fc)

            self.__setattr__(head, fc)


    def fill_fc_weights(self, m):
        if isinstance(m, nn.Conv2d):
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.base_network(x)
        # import matplotlib.pyplot as plt
        # import os
        # for idx in range(x[1].shape[1]):
        #     temp = x[1][0,idx,:,:]
        #     temp = temp.data.cpu().numpy()
        #     plt.imsave(os.path.join('dilation', '{}.png'.format(idx)), temp)
        c4_combine = self.dec_c4(x[-1], x[-2])
        c3_combine = self.dec_c3(c4_combine, x[-3]) if not self.dec_c3 == None else None
        c2_combine = self.dec_c2(c3_combine, x[-4]) if not self.dec_c2 == None else None
        c1_combine = self.dec_c1(c2_combine, x[-5]) if not self.dec_c1 == None else None  # add

        dec_dict = {}
        '''
        if not c1_combine == None:
            last_layer = c1_combine
        elif not c2_combine == None:
            last_layer = c2_combine
        elif not c3_combine == None:
            last_layer = c3_combine
        else:
            last_layer = c4_combine
        '''
        last_layer = c2_combine
        # print(last_layer.shape)
        last_layer = self.ChannelGate(last_layer)
        # print(last_layer)
        attention_map = self.DB_Blocks(last_layer)
        # print(attention_map.shape)
        soft_label = self.final_conv(attention_map).squeeze(1)
        # soft_label = torch.sigmoid(soft_label)
        # print(soft_label)
        # print(soft_label.shape)
        last_layer = last_layer * attention_map
        # print(last_layer.shape)
        for head in self.heads:
            dec_dict[head] = self.__getattr__(head)(last_layer)
            if 'hm' in head or 'cls_theta' in head:
                dec_dict[head] = torch.sigmoid(dec_dict[head])
        return dec_dict, soft_label
