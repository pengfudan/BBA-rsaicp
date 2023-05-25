import argparse
import test
import eval
from datasets.dataset_rsaicp import RSAICP
from models import ctrbox_net
import decoder
import os
import cv2
import copy
import numpy as np
from tqdm import tqdm


def rm_imgs(img_path):
    for img in os.listdir(img_path):
        imgname = os.path.join(img_path, img)
        os.remove(imgname)

def img_split(basepath, outpath, subsize, gap, rate):
    imagelist = os.listdir(basepath)
    slide = subsize - gap
    # print(imagelist)
    for imgname in tqdm(imagelist):
        img = cv2.imread(os.path.join(basepath, imgname))
        if (rate != 1):
            img = cv2.resize(img, None, fx=rate, fy=rate, interpolation = cv2.INTER_CUBIC)
        else:
            img = img
        if np.shape(img) == ():
            return
        outbasename = os.path.splitext(imgname)[0] + '__' + str(rate) + '__'
        weight = np.shape(img)[1]
        height = np.shape(img)[0]
        left, up = 0, 0
        while (left < weight):
            if (left + subsize >= weight):
                left = max(weight - subsize, 0)
            up = 0
            while (up < height):
                if (up + subsize >= height):
                    up = max(height - subsize, 0)
                subimgname = outbasename + str(left) + '___' + str(up)
                # self.f_sub.write(name + ' ' + subimgname + ' ' + str(left) + ' ' + str(up) + '\n')
                subimg = copy.deepcopy(img[up: (up + subsize), left: (left + subsize)])
                outdir = os.path.join(outpath, subimgname + '.png')
                cv2.imwrite(outdir, subimg)
                if (up + subsize >= height):
                    break
                else:
                    up = up + slide
            if (left + subsize >= weight):
                break
            else:
                left = left + slide

def parse_args():
    parser = argparse.ArgumentParser(description='BBAVectors Implementation')
    parser.add_argument('--num_epoch', type=int, default=1, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=16, help='Number of batch size')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers')
    parser.add_argument('--init_lr', type=float, default=1.25e-4, help='Initial learning rate')
    parser.add_argument('--input_h', type=int, default=608, help='Resized image height')
    parser.add_argument('--input_w', type=int, default=608, help='Resized image width')
    parser.add_argument('--K', type=int, default=500, help='Maximum of objects')
    parser.add_argument('--conf_thresh', type=float, default=0.60, help='Confidence threshold, 0.1 for general evaluation')
    parser.add_argument('--ngpus', type=int, default=4, help='Number of gpus, ngpus>1 for multigpu')
    parser.add_argument('--resume_train', type=str, default='', help='Weights resumed in training')
    parser.add_argument('--resume', type=str, default='model_rsaicp_70.pth', help='Weights resumed in testing and evaluation')
    parser.add_argument('--dataset', type=str, default='rsaicp', help='Name of dataset')
    parser.add_argument('--data_dir', type=str, default='../split_path', help='Data directory')
    parser.add_argument('--phase', type=str, default='eval', help='Phase choice= {train, test, eval}')
    parser.add_argument('--wh_channels', type=int, default=8, help='Number of channels for the vectors (4x2)')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES']='5,6'
    print('starting!')
    origin_path = '../input_path'
    split_path = '../split_path'
    if not os.path.exists(split_path):
        os.mkdir(split_path)
    # rm_imgs(split_path)
    # img_split(origin_path, split_path, subsize=608, gap=300, rate=1)
    # img_split(origin_path, split_path, subsize=608, gap=300, rate=0.8)
    # img_split(origin_path, split_path, subsize=608, gap=300, rate=0.7)
    # img_split(origin_path, split_path, subsize=608, gap=300, rate=1.3)
    # img_split(origin_path, split_path, subsize=608, gap=300, rate=1.2)

    args = parse_args()
    # dataset = {'dota': DOTA, 'hrsc': HRSC}
    dataset = {'rsaicp': RSAICP}
    # num_classes = {'dota': 15, 'hrsc': 1}
    num_classes = {'rsaicp': 11}
    heads = {'hm': num_classes[args.dataset],
             'wh': 10,
             'reg': 2,
             'cls_theta': 1
             }
    down_ratio = 4
    model = ctrbox_net.CTRBOX(heads=heads,
                              pretrained=False,
                              down_ratio=down_ratio,
                              final_kernel=1,
                              head_conv=64)

    decoder = decoder.DecDecoder(K=args.K,
                                 conf_thresh=args.conf_thresh,
                                 num_classes=num_classes[args.dataset])
    if args.phase == 'train':
        ctrbox_obj = train.TrainModule(dataset=dataset,
                                       num_classes=num_classes,
                                       model=model,
                                       decoder=decoder,
                                       down_ratio=down_ratio)

        ctrbox_obj.train_network(args)
    elif args.phase == 'test':
        ctrbox_obj = test.TestModule(dataset=dataset, num_classes=num_classes, model=model, decoder=decoder)
        ctrbox_obj.test(args, down_ratio=down_ratio)
    else:
        ctrbox_obj = eval.EvalModule(dataset=dataset, num_classes=num_classes, model=model, decoder=decoder)
        ctrbox_obj.evaluation(args, down_ratio=down_ratio)