import argparse
import train
import test
import eval
from datasets.dataset_rsaicp import RSAICP
from models import ctrbox_net
import decoder
import os


def parse_args():
    parser = argparse.ArgumentParser(description='BBAVectors Implementation')
    parser.add_argument('--num_epoch', type=int, default=1, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=1, help='Number of batch size')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers')
    parser.add_argument('--init_lr', type=float, default=2.5e-4, help='Initial learning rate')
    parser.add_argument('--input_h', type=int, default=608, help='Resized image height')
    parser.add_argument('--input_w', type=int, default=608, help='Resized image width')
    parser.add_argument('--K', type=int, default=500, help='Maximum of objects')
    parser.add_argument('--conf_thresh', type=float, default=0.28, help='Confidence threshold, 0.1 for general evaluation')
    parser.add_argument('--ngpus', type=int, default=1, help='Number of gpus, ngpus>1 for multigpu')
    parser.add_argument('--resume_train', type=str, default='', help='Weights resumed in training')
    parser.add_argument('--resume', type=str, default='model_rsaicp_80.pth', help='Weights resumed in testing and evaluation')
    parser.add_argument('--dataset', type=str, default='rsaicp', help='Name of dataset')
    parser.add_argument('--data_dir', type=str, default='./datasets/RSAICP', help='Data directory')
    parser.add_argument('--phase', type=str, default='test', help='Phase choice= {train, test, eval}')
    parser.add_argument('--wh_channels', type=int, default=8, help='Number of channels for the vectors (4x2)')
    parser.add_argument('--backbone', type=str, default="resnet18", help='Backbone for model')
    args = parser.parse_args()
    return args


def print_INFO(args):
    print("------------------------INIT------------------------")
    print("[INFO]: Machine Information:")
    print("Using gpus: {}".format(args.ngpus))
    print("number for workers: {}".format(args.num_workers))

    print("[INFO]: Image Information:")
    print("Image width: {}".format(args.input_w))
    print("Image height: {}".format(args.input_h))
    print("Max object in an image: {}".format(args.K))

    print("[INFO]: Train or Test:")
    print("{}".format(args.phase))

    if args.phase == "train":
        print("[INFO]: Model Parameters:")
        print("Num epoch: {}".format(args.num_epoch))
        print("Batch size: {}".format(args.batch_size))
        print("Init learning rate: {}".format(args.init_lr))
        print("Using data: {}, Data dir: {}".format(args.dataset, args.data_dir))
        print("Backbone: {}".format(args.backbone))
    else:
        print("Model be used: {}".format(args.model_select))
        print("Down ratio: {}".format(args.down_ratio))
        print("Resume model: {}".format(args.resume))
        print("Weather display images: {}".format(args.display))
        print("Image save dir: {}".format(args.save_dir))
        print("Confidence score threshold: {}".format(args.conf_thresh))

    print("------------------------END------------------------")


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES']='6'
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
    print_INFO(args)
    model = ctrbox_net.CTRBOX(heads=heads,
                              pretrained=True,
                              down_ratio=down_ratio,
                              final_kernel=1,
                              head_conv=64,
                              model=args.backbone)

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