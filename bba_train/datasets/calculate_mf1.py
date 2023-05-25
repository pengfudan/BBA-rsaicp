import os
import numpy as np
from .DOTA_devkit import polyiou

def parse_gt(filename):
    objects = []
    with open(filename, 'r') as f:
        for i, line in enumerate(f.readlines()):
            object_struct = {}
            obj = line.split(' ')  # list object
            if len(obj)>7:
                x1 = float(obj[0])
                y1 = float(obj[1])
                x2 = float(obj[2])
                y2 = float(obj[3])
                x3 = float(obj[4])
                y3 = float(obj[5])
                x4 = float(obj[6])
                y4 = float(obj[7])
                # TODO: filter small instances
                xmin = max(min(x1, x2, x3, x4), 0)
                xmax = max(x1, x2, x3, x4)
                ymin = max(min(y1, y2, y3, y4), 0)
                ymax = max(y1, y2, y3, y4)
                if ((xmax - xmin) > 10) and ((ymax - ymin) > 10):
                    object_struct['name'] = obj[8][0]
                    object_struct['bbox'] = [x1, y1, x2, y2, x3, y3, x4, y4]
                    objects.append(object_struct)
    return objects


def cal_rec_prec(detpath,
                 annopath,
                 imagesetfile,
                 classname,
                 ovthresh=0.5):

    with open(imagesetfile, 'r') as f:
        lines = f.readlines()
    
    imagenames = [x.strip() for x in lines]
    recs = {}
    for i, imagename in enumerate(imagenames):
        recs[imagename] = parse_gt(os.path.join(annopath.format(imagename)))
    # extract gt objects for this class
    class_recs = {}
    npos = 0
    for imagename in imagenames:
        R = [obj for obj in recs[imagename] if obj['name'] == classname]
        bbox = np.array([x['bbox'] for x in R])
        det = [False] * len(R)
        npos = npos + len([x['name'] for x in R])  # num of gt
        class_recs[imagename] = {'bbox': bbox,
                                 'det': det}

    # read dets from Task1* files
    detfile = detpath.format(classname)
    with open(detfile, 'r') as f:
        lines = f.readlines()

    splitlines = [x.strip().split(' ') for x in lines]
    image_ids = [x[0] for x in splitlines]
    confidence = np.array([float(x[1]) for x in splitlines])

    #print('check confidence: ', confidence)

    BB = np.array([[float(z) for z in x[2:]] for x in splitlines])

    if len(confidence)>1:
        # sort by confidence
        sorted_ind = np.argsort(-confidence)
        sorted_scores = np.sort(-confidence)

        #print('check sorted_scores: ', sorted_scores)
        #print('check sorted_ind: ', sorted_ind)

        ## note the usage only in numpy not for list
        BB = BB[sorted_ind, :]
        image_ids = [image_ids[x] for x in sorted_ind]
    #print('check imge_ids: ', image_ids)
    #print('imge_ids len:', len(image_ids))
    # go down dets and mark TPs and FPs
    nd = len(image_ids)
    tp = np.zeros(nd)
    fp = np.zeros(nd)
    for d in range(nd):
        R = class_recs[image_ids[d]]
        bb = BB[d, :].astype(float)
        ovmax = -np.inf
        BBGT = R['bbox'].astype(float)

        ## compute det bb with each BBGT

        if BBGT.size > 0:
            # compute overlaps
            # intersection

            # 1. calculate the overlaps between hbbs, if the iou between hbbs are 0, the iou between obbs are 0, too.
            # pdb.set_trace()
            BBGT_xmin =  np.min(BBGT[:, 0::2], axis=1)
            BBGT_ymin = np.min(BBGT[:, 1::2], axis=1)
            BBGT_xmax = np.max(BBGT[:, 0::2], axis=1)
            BBGT_ymax = np.max(BBGT[:, 1::2], axis=1)
            bb_xmin = np.min(bb[0::2])
            bb_ymin = np.min(bb[1::2])
            bb_xmax = np.max(bb[0::2])
            bb_ymax = np.max(bb[1::2])

            ixmin = np.maximum(BBGT_xmin, bb_xmin)
            iymin = np.maximum(BBGT_ymin, bb_ymin)
            ixmax = np.minimum(BBGT_xmax, bb_xmax)
            iymax = np.minimum(BBGT_ymax, bb_ymax)
            iw = np.maximum(ixmax - ixmin + 1., 0.)
            ih = np.maximum(iymax - iymin + 1., 0.)
            inters = iw * ih

            # union
            uni = ((bb_xmax - bb_xmin + 1.) * (bb_ymax - bb_ymin + 1.) +
                   (BBGT_xmax - BBGT_xmin + 1.) *
                   (BBGT_ymax - BBGT_ymin + 1.) - inters)

            overlaps = inters / uni

            BBGT_keep_mask = overlaps > 0
            BBGT_keep = BBGT[BBGT_keep_mask, :]
            BBGT_keep_index = np.where(overlaps > 0)[0]
            # pdb.set_trace()
            def calcoverlaps(BBGT_keep, bb):
                overlaps = []
                for index, GT in enumerate(BBGT_keep):

                    overlap = polyiou.iou_poly(polyiou.VectorDouble(BBGT_keep[index]), polyiou.VectorDouble(bb))
                    overlaps.append(overlap)
                return overlaps
            if len(BBGT_keep) > 0:
                overlaps = calcoverlaps(BBGT_keep, bb)

                ovmax = np.max(overlaps)
                jmax = np.argmax(overlaps)
                # pdb.set_trace()
                jmax = BBGT_keep_index[jmax]

        if ovmax > ovthresh:
            if not R['det'][jmax]:
                tp[d] = 1.
                R['det'][jmax] = 1
            else:
                fp[d] = 1.
        else:
            fp[d] = 1.

    # compute precision recall

    # print('check fp:', fp)
    # print('check tp', tp)


    # print('npos num:', npos)
    fp = np.sum(fp)
    tp = np.sum(tp)

    rec = tp / float(npos)
    # avoid divide by zero in case the first detection matches a difficult
    # ground truth
    prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)

    return rec, prec


if __name__ == '__main__':
    category = ['A','B','C','D','E','F','G','H','I','J','K']
    data_dir = './datasets/RSAICP'
    detpath = os.path.join(result_path, 'Task_{}.txt')
    annopath = os.path.join('/data3/plin/rsaicp/val', '{}.txt')
    imagesetfile = os.path.join(self.data_dir, 'val.txt')
    rec, prec = cal_rec_prec(detpath, annopath, imagesetfile, classname, ovthresh=0.5)
    print('A:Precision: 0 Recall: 0')
    