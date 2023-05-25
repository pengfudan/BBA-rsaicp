"""
-------------
This is the multi-process version
"""
import os
import codecs
import numpy as np
import math
from dota_utils_mine import GetFileFromThisRootDir
import cv2
import shapely.geometry as shgeo
import dota_utils_mine as util
import copy
from multiprocessing import Pool
from functools import partial
import time

GAP = 200
SUBSIZE = 1024

def point2theta(axis):
    '''
        input: np.array[x axis, y axis] * 4; cnt
        output: list[x, y , l, s, theta] //l:Long side; s:Short side
        4 points -> openCV表示�?-> 长边表示�?    '''
    # point1 = np.array(shape['points'][0])
    # point2 = np.array(shape['points'][1])
    # point3 = np.array(shape['points'][2])
    # point4 = np.array(shape['points'][3])
    cnt =  np.array(axis,dtype='float32').reshape((4,2))

    rect = cv2.minAreaRect(cnt)# 得到label最小外接矩形的（中�?x,y), (�?�?, 旋转角度�?    
    c_x = rect[0][0]
    c_y = rect[0][1]
    w = rect[1][0]
    h = rect[1][1]
    theta = rect[-1]

    box = cv2.boxPoints(rect)

    l = h
    s = w

    #theta[-90, 0)->[0, 180)
    if theta == 0 and w < h:
        theta = -90
        l = h
        s = w

    if w > h:
        l = w
        s = h
    else:
        if theta == 0:
            theta = 0
        else:
            theta = 90 + theta

    loc = [c_x, c_y, l, s, theta]
    norm = float(SUBSIZE)
    norm_loc = [round(c_x/norm,4), round(c_y/norm,4,), round(l/norm,4), round(s/norm,4), int(theta)]
    return norm_loc

def choose_best_pointorder_fit_another(poly1, poly2):
    """
        To make the two polygons best fit with each point
    """
    x1 = poly1[0]
    y1 = poly1[1]
    x2 = poly1[2]
    y2 = poly1[3]
    x3 = poly1[4]
    y3 = poly1[5]
    x4 = poly1[6]
    y4 = poly1[7]
    combinate = [np.array([x1, y1, x2, y2, x3, y3, x4, y4]), np.array([x2, y2, x3, y3, x4, y4, x1, y1]),
                 np.array([x3, y3, x4, y4, x1, y1, x2, y2]), np.array([x4, y4, x1, y1, x2, y2, x3, y3])]
    dst_coordinate = np.array(poly2)
    distances = np.array([np.sum((coord - dst_coordinate)**2) for coord in combinate])
    sorted = distances.argsort()
    return combinate[sorted[0]]

def cal_line_length(point1, point2):
    return math.sqrt( math.pow(point1[0] - point2[0], 2) + math.pow(point1[1] - point2[1], 2))


def split_single_warp(name, split_base, rate, extent):
    split_base.SplitSingle(name, rate, extent)

class splitbase():
    def __init__(self,
                 basepath,
                 outpath,
                 code = 'utf-8',
                 gap=512,
                 subsize=1024,
                 thresh=0.7,
                 choosebestpoint=True,
                 ext = '.png',
                 padding=True,
                 num_process=8
                 ):
        """
        :param basepath: base path for dota data
        :param outpath: output base path for dota data,
        the basepath and outputpath have the similar subdirectory, 'images' and 'labelTxt'
        :param code: encodeing format of txt file
        :param gap: overlap between two patches
        :param subsize: subsize of patch
        :param thresh: the thresh determine whether to keep the instance if the instance is cut down in the process of split
        :param choosebestpoint: used to choose the first point for the
        :param ext: ext for the image format
        :param padding: if to padding the images so that all the images have the same size
        """
        self.basepath = basepath
        self.outpath = outpath
        self.code = code
        self.gap = gap
        self.subsize = subsize
        self.slide = self.subsize - self.gap
        self.thresh = thresh
        self.imagepath = os.path.join(self.basepath, 'images')
        self.labelpath = os.path.join(self.basepath, 'labelTxt')
        self.outimagepath = os.path.join(self.outpath, 'images')
        self.outlabelpath = os.path.join(self.outpath, 'labelTxt')
        self.choosebestpoint = choosebestpoint
        self.ext = ext
        self.padding = padding
        self.num_process = num_process
        self.pool = Pool(num_process)
        print('padding:', padding)

        # pdb.set_trace()
        if not os.path.isdir(self.outpath):
            os.mkdir(self.outpath)
        if not os.path.isdir(self.outimagepath):
            # pdb.set_trace()
            os.mkdir(self.outimagepath)
        if not os.path.isdir(self.outlabelpath):
            os.mkdir(self.outlabelpath)
        # pdb.set_trace()
    ## point: (x, y), rec: (xmin, ymin, xmax, ymax)
    # def __del__(self):
    #     self.f_sub.close()
    ## grid --> (x, y) position of grids
    def polyorig2sub(self, left, up, poly):
        polyInsub = np.zeros(len(poly))
        for i in range(int(len(poly)/2)):
            polyInsub[i * 2] = int(poly[i * 2] - left)
            polyInsub[i * 2 + 1] = int(poly[i * 2 + 1] - up)
        return polyInsub

    def calchalf_iou(self, poly1, poly2):
        """
            It is not the iou on usual, the iou is the value of intersection over poly1
        """
        inter_poly = poly1.intersection(poly2)
        inter_area = inter_poly.area
        poly1_area = poly1.area
        half_iou = inter_area / poly1_area
        return inter_poly, half_iou

    def saveimagepatches(self, img, subimgname, left, up):
        subimg = copy.deepcopy(img[up: (up + self.subsize), left: (left + self.subsize)])
        outdir = os.path.join(self.outimagepath, subimgname + self.ext)
        h, w, c = np.shape(subimg)
        if (self.padding):
            outimg = np.zeros((self.subsize, self.subsize, 3))
            outimg[0:h, 0:w, :] = subimg
            cv2.imwrite(outdir, outimg)
        else:
            cv2.imwrite(outdir, subimg)

    def GetPoly4FromPoly5(self, poly):
        distances = [cal_line_length((poly[i * 2], poly[i * 2 + 1] ), (poly[(i + 1) * 2], poly[(i + 1) * 2 + 1])) for i in range(int(len(poly)/2 - 1))]
        distances.append(cal_line_length((poly[0], poly[1]), (poly[8], poly[9])))
        pos = np.array(distances).argsort()[0]
        count = 0
        outpoly = []
        while count < 5:
            #print('count:', count)
            if (count == pos):
                outpoly.append((poly[count * 2] + poly[(count * 2 + 2)%10])/2)
                outpoly.append((poly[(count * 2 + 1)%10] + poly[(count * 2 + 3)%10])/2)
                count = count + 1
            elif (count == (pos + 1)%5):
                count = count + 1
                continue

            else:
                outpoly.append(poly[count * 2])
                outpoly.append(poly[count * 2 + 1])
                count = count + 1
        return outpoly

    def savepatches(self, resizeimg, objects, subimgname, left, up, right, down):
        outdir = os.path.join(self.outlabelpath, subimgname + '.txt')
        mask_poly = []
        imgpoly = shgeo.Polygon([(left, up), (right, up), (right, down),
                                 (left, down)])
        with codecs.open(outdir, 'w', self.code) as f_out:
            for obj in objects:
                gtpoly = shgeo.Polygon([(obj['poly'][0], obj['poly'][1]),
                                         (obj['poly'][2], obj['poly'][3]),
                                         (obj['poly'][4], obj['poly'][5]),
                                         (obj['poly'][6], obj['poly'][7])])
                if (gtpoly.area <= 0):
                    continue
                inter_poly, half_iou = self.calchalf_iou(gtpoly, imgpoly)

                if (half_iou == 1):
                    polyInsub = self.polyorig2sub(left, up, obj['poly'])
                    
                    # polyInsub = point2theta(polyInsub)
                    
                    outline = ' '.join(list(map(str, polyInsub)))
                    outline = obj['cls'] + ' ' + outline
                    f_out.write(outline + '\n')
                
                elif (half_iou > 0):
                    inter_poly = shgeo.polygon.orient(inter_poly, sign=1)
                    out_poly = list(inter_poly.exterior.coords)[0: -1]
                    if len(out_poly) < 4:
                        continue

                    out_poly2 = []
                    for i in range(len(out_poly)):
                        out_poly2.append(out_poly[i][0])
                        out_poly2.append(out_poly[i][1])

                    if (len(out_poly) == 5):
                        out_poly2 = self.GetPoly4FromPoly5(out_poly2)
                    elif (len(out_poly) > 5):
                        """
                            if the cut instance is a polygon with points more than 5, we do not handle it currently
                        """
                        continue
                    if (self.choosebestpoint):
                        out_poly2 = choose_best_pointorder_fit_another(out_poly2, obj['poly'])

                    polyInsub = self.polyorig2sub(left, up, out_poly2)
                    # polyInsub = point2theta(polyInsub)

                    for index, item in enumerate(polyInsub):
                        # if (item <= 1):
                        #     polyInsub[index] = 1
                        #TODO:
                        if (item >= self.subsize):
                            polyInsub[index] = 1
                    
                    outline = ' '.join(list(map(str, polyInsub)))
                    outline = obj['cls'] + ' ' + outline
                    f_out.write(outline + '\n')

        self.saveimagepatches(resizeimg, subimgname, left, up)

    def SplitSingle(self, name, rate, extent):
        """
            split a single image and ground truth
        :param name: image name
        :param rate: the resize scale for the image
        :param extent: the image format
        :return:
        """
        img = cv2.imread(os.path.join(self.imagepath, name + extent))
        if np.shape(img) == ():
            return
        fullname = os.path.join(self.labelpath, name + '.txt')
        objects = util.parse_dota_poly2(fullname)
        for obj in objects:
            obj['poly'] = list(map(lambda x:rate*x, obj['poly']))
            #obj['poly'] = list(map(lambda x: ([2 * y for y in x]), obj['poly']))

        if (rate != 1):
            resizeimg = cv2.resize(img, None, fx=rate, fy=rate, interpolation = cv2.INTER_CUBIC)
        else:
            resizeimg = img
        outbasename = name + '_' + str(rate) + '_'
        weight = np.shape(resizeimg)[1]
        height = np.shape(resizeimg)[0]

        left, up = 0, 0
        while (left < weight):
            if (left + self.subsize >= weight):
                left = max(weight - self.subsize, 0)
            up = 0
            while (up < height):
                if (up + self.subsize >= height):
                    up = max(height - self.subsize, 0)
                right = min(left + self.subsize, weight - 1)
                down = min(up + self.subsize, height - 1)
                subimgname = outbasename + str(left) + '__' + str(up)
                # self.f_sub.write(name + ' ' + subimgname + ' ' + str(left) + ' ' + str(up) + '\n')
                self.savepatches(resizeimg, objects, subimgname, left, up, right, down)
                if (up + self.subsize >= height):
                    break
                else:
                    up = up + self.slide
            if (left + self.subsize >= weight):
                break
            else:
                left = left + self.slide

    def splitdata(self, rate):
        """
        :param rate: resize rate before cut
        """
        imagelist = GetFileFromThisRootDir(self.imagepath)
        imagenames = [util.custombasename(x) for x in imagelist if (util.custombasename(x) != 'Thumbs')]
        if self.num_process == 1:
            for name in imagenames:
                self.SplitSingle(name, rate, self.ext)
        else:

            # worker = partial(self.SplitSingle, rate=rate, extent=self.ext)
            worker = partial(split_single_warp, split_base=self, rate=rate, extent=self.ext)
            self.pool.map(worker, imagenames)

    def __getstate__(self):
        self_dict = self.__dict__.copy()
        del self_dict['pool']
        return self_dict

    def __setstate__(self, state):
        self.__dict__.update(state)

if __name__ == '__main__':
    split = splitbase(r'/data3/syfeng/MyProject/rsaicp/tools/data/fulldata/val',
                       r'/data3/syfeng/MyProject/rsaicp/tools/data/splitdata/val',
                      gap=GAP,
                      subsize=SUBSIZE,
                      num_process=8
                      )
    #裁剪前对图像进行rate倍的resize，该参数用于离线多尺度裁�?    split.splitdata(1)

'''
2_1_0__158.png 图片是将原图 P0706.png resize为原来的1倍，在width=xxx，height=xxx处进行裁剪�?该位置信息在merge检测结果这一步骤中至关重要，因此不要更改切割后的图片以及label文件的文件名称�?split = splitbase(
                    basepath='待分割数据集文件路径',
                    outpath='分割后的数据及文件保存路�?
                    gap=两张被分割图片之间的重叠区域,
                    subsize=分割后的图片size,
                    thresh=如果实例在拆分过程中被截断，thresh决定是否保留实例，默�?.7
           )
split.splitdata(rate='裁剪前对图像进行比例resize，该参数用于离线多尺度裁�?)

注意：不规则四边形在裁剪图像过程中有概率会被截断，是否保留该目标参考以下几种情况：
      1. 被截断后的物体包络框顶点数小�?，该目标不保�?      2. 被截断后的物体包络框顶点数大�?，该目标不保�?      3. 被截断后的物体包络框与原始目标边框重叠区域占比超过thresh时，正常保留目标
      4. �?种情况中重叠区域占比低于thresh时，目标的diffcult设为2，即更难识别的目�?'''