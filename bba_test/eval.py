import torch
import os
import func_utils
import json


class EvalModule(object):
    def __init__(self, dataset, num_classes, model, decoder):
        torch.manual_seed(317)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.dataset = dataset
        self.num_classes = num_classes
        self.model = model
        self.decoder = decoder


    def load_model(self, model, resume):
        checkpoint = torch.load(resume, map_location=lambda storage, loc: storage)
        print('loaded weights from {}, epoch {}'.format(resume, checkpoint['epoch']))
        state_dict_ = checkpoint['model_state_dict']
        model.load_state_dict(state_dict_, strict=False)
        return model

    def evaluation(self, args, down_ratio):
        save_path = 'weights_'+args.dataset
        self.model = self.load_model(self.model, os.path.join(save_path, args.resume))
        self.model = self.model.to(self.device)
        self.model.eval()

        '''
        result_path = 'result_'+args.dataset
        if not os.path.exists(result_path):
            os.mkdir(result_path)
        '''
        dataset_module = self.dataset[args.dataset]
        dsets = dataset_module(data_dir=args.data_dir,
                               phase='test',
                               input_h=args.input_h,
                               input_w=args.input_w,
                               down_ratio=down_ratio)
        '''
        func_utils.write_results(args,
                                 self.model,
                                 dsets,
                                 down_ratio,
                                 self.device,
                                 self.decoder,
                                 result_path,
                                 print_ps=False)
        if args.dataset == 'rsaicp':
            merge_path = 'merge_'+args.dataset
            if not os.path.exists(merge_path):
                os.mkdir(merge_path)
            dsets.merge_crop_image_results(result_path, merge_path)
        '''
        merge_path = 'merge_'+args.dataset
        if args.dataset == 'rsaicp':
            final_path = 'final_'+args.dataset
            if not os.path.exists(final_path):
                os.mkdir(final_path)
            dsets.merge_tiff_image_results(merge_path, final_path)
        
        final_path = 'final_'+args.dataset
        final_txt = '../final_txt'
        if not os.path.exists(final_txt):
            os.mkdir(final_txt)
        # category = ['A','B','C','D','E','F','G','H','I','J','K']
        category = ['A','B']
        counts = []
        for cat in category:
            filename = os.path.join(final_path, 'Task_{}.txt'.format(cat))
            count = 0
            with open(filename, 'r') as f:
                for line in f.readlines():
                    count = count + 1
                    object_dict = {}
                    splitlines = line.strip().split(' ')  # list object
                    with open(os.path.join(final_txt, splitlines[0] + ".txt"), 'a') as f1: #  此处是追加！！！
                        f1.write('{} {:.12f} {:.1f} {:.1f} {:.1f} {:.1f} {:.1f} {:.1f} {:.1f} {:.1f}\n'.format(
                            cat, float(splitlines[1]), float(splitlines[2]), float(splitlines[3]), float(splitlines[4]), float(splitlines[5]), float(splitlines[6]), float(splitlines[7]), float(splitlines[8]), float(splitlines[9])))
                    f1.close()
            counts.append(count)
        print(counts)
        