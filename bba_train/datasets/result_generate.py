import cv2
import os
import json
import numpy as np

        
if __name__ == '__main__':
    json_path = '/data3/plin/output_path/aircraft_results.json'
    result_path = '/data3/plin/result_json'
    with open(json_path, 'r') as load_f:
        load_list = json.load(load_f)
    # print(load_list[0])
    if not os.path.exists(result_path):
        os.mkdir(result_path)
    for det in load_list:
        labelme_dict = {}
        labelme_dict["version"] = "4.5.6"
        labelme_dict["flags"] = {}
        det_list = []
        for object in det['labels']:
            det_object = {}
            det_object["label"] = object['category_id']
            det_object["points"] = object['points']
            det_object["group_id"] = None
            det_object["shape_type"] = "polygon"
            det_object["flags"] = {}
            det_list.append(det_object)
        labelme_dict["shapes"] = det_list
        # labelme_dict["imagePath"] = os.path.join('/val', det['image_name'])
        labelme_dict["imagePath"] = det['image_name']
        labelme_dict["imageHeight"] = 4096
        labelme_dict["imageWidth"] = 4096
        labelme_dict["imageData"] = None
        json_name = os.path.splitext(det['image_name'])[0] + '.json'
        with open(os.path.join(result_path, json_name), "w") as dump_f:
            json.dump(labelme_dict, dump_f, indent=4)

