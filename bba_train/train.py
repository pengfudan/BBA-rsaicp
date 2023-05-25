import torch
import torch.nn as nn
import os
import numpy as np
import loss
import cv2
import func_utils
from tqdm import tqdm


def collater(data):
    out_data_dict = {}
    for name in data[0]:
        out_data_dict[name] = []
    for sample in data:
        for name in sample:
            out_data_dict[name].append(torch.from_numpy(sample[name]))
    for name in out_data_dict:
        out_data_dict[name] = torch.stack(out_data_dict[name], dim=0)
    return out_data_dict

class TrainModule(object):
    def __init__(self, dataset, num_classes, model, decoder, down_ratio):
        torch.manual_seed(317)
        self.dataset = dataset
        self.dataset_phase = {'rsaicp': ['train']}
        self.num_classes = num_classes
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = model
        self.decoder = decoder
        self.down_ratio = down_ratio

    def save_model(self, path, epoch, model, optimizer):
        if isinstance(model, torch.nn.DataParallel):
            state_dict = model.module.state_dict()
        else:
            state_dict = model.state_dict()
        torch.save({
            'epoch': epoch,
            'model_state_dict': state_dict,
            'optimizer_state_dict': optimizer.state_dict(),
            # 'loss': loss
        }, path)

    def load_model(self, model, optimizer, resume, strict=True):
        checkpoint = torch.load(resume, map_location=lambda storage, loc: storage)
        print('loaded weights from {}, epoch {}'.format(resume, checkpoint['epoch']))
        state_dict_ = checkpoint['model_state_dict']
        state_dict = {}
        for k in state_dict_:
            if k.startswith('module') and not k.startswith('module_list'):
                state_dict[k[7:]] = state_dict_[k]
            else:
                state_dict[k] = state_dict_[k]
        model_state_dict = model.state_dict()
        if not strict:
            for k in state_dict:
                if k in model_state_dict:
                    if state_dict[k].shape != model_state_dict[k].shape:
                        print('Skip loading parameter {}, required shape{}, ' \
                              'loaded shape{}.'.format(k, model_state_dict[k].shape, state_dict[k].shape))
                        state_dict[k] = model_state_dict[k]
                else:
                    print('Drop parameter {}.'.format(k))
            for k in model_state_dict:
                if not (k in state_dict):
                    print('No param {}.'.format(k))
                    state_dict[k] = model_state_dict[k]
        model.load_state_dict(state_dict, strict=False)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.cuda()
        epoch = checkpoint['epoch']
        # loss = checkpoint['loss']
        return model, optimizer, epoch

    def train_network(self, args):

        self.optimizer = torch.optim.Adam(self.model.parameters(), args.init_lr)
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.96, last_epoch=-1)
        #save_path1 = 'weights_'+args.dataset+'_ms3_res50_dow4'  # modify
        save_path = 'weights_'+args.dataset+'_res18_ms3_emfl_att_b16_25e4'  # modify
        start_epoch = 1
        # add resume part for continuing training when break previously, 10-16-2020
        if args.resume_train:
            self.model, self.optimizer, start_epoch = self.load_model(self.model, 
                                                                        self.optimizer, 
                                                                        args.resume_train, 
                                                                        strict=True)
        # end 

        if not os.path.exists(save_path):
            os.mkdir(save_path)
        if args.ngpus>1:
            if torch.cuda.device_count() > 1:
                print("Let's use", torch.cuda.device_count(), "GPUs!")
                # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
                self.model = nn.DataParallel(self.model)
        self.model.to(self.device)

        criterion = loss.LossAll()
        print('Setting up data...')

        dataset_module = self.dataset[args.dataset]

        dsets = {x: dataset_module(data_dir=args.data_dir,
                                   phase=x,
                                   input_h=args.input_h,
                                   input_w=args.input_w,
                                   down_ratio=self.down_ratio)
                 for x in self.dataset_phase[args.dataset]}

        dsets_loader = {}
        dsets_loader['train'] = torch.utils.data.DataLoader(dsets['train'],
                                                           batch_size=args.batch_size,
                                                           shuffle=True,
                                                           num_workers=args.num_workers,
                                                           pin_memory=True,
                                                           drop_last=True,
                                                           collate_fn=collater)

        print('Starting training...')
        train_loss = []
        ap_list = []
        for epoch in range(start_epoch, args.num_epoch+1):
            print('-'*10)
            print('Epoch: {}/{} '.format(epoch, args.num_epoch))
            epoch_loss = self.run_epoch(phase='train',
                                        data_loader=dsets_loader['train'],
                                        criterion=criterion,
                                        epoch=epoch)
            train_loss.append(epoch_loss)
            self.scheduler.step(epoch)

            np.savetxt(os.path.join(save_path, 'train_loss.txt'), train_loss, fmt='%.6f')

            if epoch % 10 == 0 or epoch > 140:
                self.save_model(os.path.join(save_path, 'model_rsaicp_{}.pth'.format(epoch)),
                                epoch,
                                self.model,
                                self.optimizer)

            

            if 'test' in self.dataset_phase[args.dataset] and epoch % 10==0:
                mAP = self.dec_eval(args, dsets['test'])
                ap_list.append(mAP)
                np.savetxt(os.path.join(save_path, 'ap_list.txt'), ap_list, fmt='%.6f')

            self.save_model(os.path.join(save_path, 'model_rsaicp_last.pth'),
                            epoch,
                            self.model,
                            self.optimizer)

    def run_epoch(self, phase, data_loader, criterion, epoch):
        if phase == 'train':
            self.model.train()
        else:
            self.model.eval()
        running_loss = 0.
        # running_cl_loss = 0.
        running_hm_loss = 0.
        # running_cc_loss = 0.
        running_wh_loss = 0.
        running_off_loss = 0.
        running_cls_theta_loss = 0.
        running_attention_loss = 0.
        for data_dict in tqdm(data_loader):
            for name in data_dict:
                data_dict[name] = data_dict[name].to(device=self.device, non_blocking=True)
            if phase == 'train':
                self.optimizer.zero_grad()
                with torch.enable_grad():
                    pr_decs, soft_label = self.model(data_dict['input'])  # add feat
                    loss, hm_loss, wh_loss, off_loss, cls_theta_loss, attention_loss = criterion(pr_decs, soft_label, data_dict)
                    loss.backward()
                    self.optimizer.step()
            else:
                with torch.no_grad():
                    pr_decs, soft_label = self.model(data_dict['input'])
                    loss, hm_loss, wh_loss, off_loss, cls_theta_loss, attention_loss = criterion(pr_decs, soft_label, data_dict)

            running_loss += loss.item()
            # running_cl_loss += cl_loss.item()
            # running_cc_loss += cc_loss.item()
            running_hm_loss += hm_loss.item()
            running_wh_loss += wh_loss.item()
            running_off_loss += off_loss.item()
            running_cls_theta_loss += cls_theta_loss.item()
            running_attention_loss += attention_loss.item()
        epoch_loss = running_loss / len(data_loader)
        print('{} loss: {}'.format(phase, epoch_loss))
        # print('cl loss: {}'.format(running_cl_loss / len(data_loader)))
        # print('cc loss: {}'.format(running_cc_loss / len(data_loader)))
        print('hm loss: {}'.format(running_hm_loss / len(data_loader)))
        print('wh loss: {}'.format(running_wh_loss / len(data_loader)))
        print('off loss: {}'.format(running_off_loss / len(data_loader)))
        print('cls_theta loss: {}'.format(running_cls_theta_loss / len(data_loader)))
        print('attention loss: {}'.format(running_attention_loss / len(data_loader)))
        return epoch_loss


    def dec_eval(self, args, dsets):
        result_path = 'result_'+args.dataset
        if not os.path.exists(result_path):
            os.mkdir(result_path)

        self.model.eval()
        func_utils.write_results(args,
                                 self.model,dsets,
                                 self.down_ratio,
                                 self.device,
                                 self.decoder,
                                 result_path)
        ap = dsets.dec_evaluation(result_path)
        return ap