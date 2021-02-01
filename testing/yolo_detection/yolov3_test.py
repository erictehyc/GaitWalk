import os
import cv2
import time
import argparse
import torch
import warnings
import numpy as np
import sys
from pathlib import Path
base_path = os.getcwd()
while os.path.basename(base_path) != "fyp_team4c":
    path = Path(base_path)
    base_path = str(path.parent)
    if base_path == '/':
        print("Please call this script in the fyp_team4c directory")
        break
deepsort_path = os.path.join(base_path, 'deep_sort_pytorch')
test_path = os.path.join(base_path, 'unit_testing/yolo_detection')
sys.path.append(deepsort_path)
os.chdir(deepsort_path)
print(os.getcwd())
from detector import build_detector
from utils.draw import draw_boxes
from utils.parser import get_config

def write_bbox_to_txtfile(test_img_dir, show_image=False):
    cfg = get_config()
    cfg.merge_from_file(f'{deepsort_path}/configs/yolov3.yaml')
    cfg.merge_from_file(f'{deepsort_path}/configs/deep_sort.yaml')
    use_cuda = torch.cuda.is_available()
    detector = build_detector(cfg, use_cuda=use_cuda)

    img_file_l = [fn for fn in os.listdir(test_img_dir) if fn.endswith(".jpg")]
    for img_fn in img_file_l:
        print("Detecting ", img_fn)
        res = []
        img_path = os.path.join(test_img_dir, img_fn)
        ori_im = cv2.imread(img_path)
        im = cv2.cvtColor(ori_im, cv2.COLOR_BGR2RGB)
        (im_h, im_w,_) = im.shape
        bbox_xywh, cls_conf, cls_ids = detector(im)

        for i, cls_id in enumerate(cls_ids):
            if cls_id != 0:
                continue
            #only adds bounding box for "person" detected
            x,y,w,h = bbox_xywh[i]
            x1,y1,x2,y2 = int(x-w//2),int(y-h//2),int(x+w//2),int(y+h//2)
            if show_image:
                print("drawing rectangle")
                #draw bounding box on image
                cv2.rectangle(ori_im,(x1,y1),(x2,y2),(0,255,0),6)

            x = max(0, x/im_w)
            y = max(0, y/im_h)
            w = min(w/im_w, 1)
            h = min(h/im_h, 1)
            # print([cls_ids[i], cls_conf[i], x1, y1, x2, y2])
            res.append(f"{cls_ids[i]} {cls_conf[i]} {x} {y} {w} {h}")

        if show_image:
            res_img_path = os.path.join(test_path, "result_image", img_fn[:-4]+'.jpg')
            print("writing to: ", res_img_path)
            status=cv2.imwrite(res_img_path, ori_im)
            print("Status: ", status)

        res_s = '\n'.join(res)     
        with open(f'{test_path}/detection-results/{img_fn[:-4]}.txt', 'w+') as f:
            f.write(res_s)
        

    # print(bbox_xywh, cls_conf, cls_ids)

if __name__ == "__main__":
    test_img_dir = os.path.join(test_path, 'test_images/')
    write_bbox_to_txtfile(test_img_dir, show_image=True)
