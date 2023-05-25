import cv2
import os
import json
import numpy as np
from tqdm import tqdm

dir_json = '/data3/plin/rsaicpfinal/train/labelJson/'
dir_txt = '/data3/plin/rsaicpfinal/train/labelTxt/'
if not os.path.exists(dir_txt):
    os.mkdir(dir_txt)

def get_files_list(path):
    
    json_files_list = []
    for pos_json in os.listdir(path):
        if pos_json.endswith('.json'):
            json_files_list.append(pos_json)

    return json_files_list

def point2theta(cnt):
    rect = cv2.minAreaRect(cnt)
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
    return loc

def json2txt(json_files_list, dir_json, dir_txt):
    
    for json_file in tqdm(json_files_list):
        path_json = os.path.join(dir_json, json_file)
        # index = ''.join(list(filter(str.isdigit, json_file)))
        # path_txt = os.path.join(dir_txt, index + '.txt')
        path_txt = os.path.join(dir_txt, json_file.replace('.json', '.txt'))

        with open(path_json, 'r') as path_json:
            jsonx = json.load(path_json)

            with open(path_txt, 'w+') as ftxt:
                for shape in jsonx['shapes']:
                    loc = np.array(shape['points'],dtype='float32')
                    
                    label = str(loc[0][0])+' '+str(loc[0][1])+' '+str(loc[1][0])+' '+str(loc[1][1])+' '+\
                             str(loc[2][0])+' '+str(loc[2][1])+' '+str(loc[3][0])+' '+str(loc[3][1])+' '
                    
                    label += shape['label']

                    ftxt.writelines(label+'\n')
        
if __name__ == '__main__':
    
    json_files_list  = get_files_list(dir_json)
    json2txt(json_files_list, dir_json, dir_txt)

