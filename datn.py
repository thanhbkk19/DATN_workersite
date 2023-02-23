import os
from re import T
import sys
import argparse
import os

# limit the number of cpus used by high performance libraries
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import time
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random
import numpy as np
from random import random as ran
from datetime import datetime
import time
import pandas as pd
from collections import Counter
import threading
from threading import Thread
import warnings
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')


FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # strongsort root directory
WEIGHTS = ROOT / 'weights'


if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
if str(ROOT / 'yolov7') not in sys.path:
    sys.path.append(str(ROOT / 'yolov7'))  # add yolov7 ROOT to PATH
if str(ROOT / 'strong_sort') not in sys.path:
    sys.path.append(str(ROOT / 'strong_sort'))  # add strong_sort ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative


from yolov7.models.experimental import attempt_load
from yolov7.utils.datasets import LoadStreams, LoadImages
from yolov7.utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from yolov7.utils.plots import plot_one_box
from yolov7.utils.torch_utils import select_device, load_classifier, time_synchronized, TracedModel

from strong_sort.utils.parser import get_config
from strong_sort.strong_sort import StrongSORT

def detect(line_thickness=1):
    source, weights, show_vid, save_txt, imgsz = '638229838.mp4', 'best.pt', True, True, 720
    webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
        ('rtsp://', 'rtmp://', 'http://', 'https://'))

    strong_sort_weights=WEIGHTS / 'osnet_x0_25_msmt17.pt'  # model.pt path,
    config_strongsort=ROOT / 'strong_sort/configs/strong_sort.yaml'
    count= False
    line_thickness=opt.line_thickness

    
    # Directories
    # Initialize
    set_logging()
    device = select_device(opt.device)
    half = True  # half precision only supported on CUDA

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size

    if half:
        model.half()  # to FP16
    list_names_cam = []
    # Set Dataloader
    vid_path, vid_writer = None, None
    if webcam:
        view_img = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride)
        nr_sources = len(dataset)
        nr_sources = 2
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride)
        nr_sources = 1
    vid_path, vid_writer, txt_path = [None] * nr_sources, [None] * nr_sources, [None] * nr_sources

    if nr_sources >=2:
        list_names_cam.extend(['Camera ' + str(i) for i in range(1,nr_sources+1)])

    # initialize StrongSORT
    cfg = get_config()
    cfg.merge_from_file('strong_sort/configs/strong_sort.yaml')

    # Create as many strong sort instances as there are video sources
    strongsort_list = []
    num_person = 0
    for i in range(nr_sources):
        strongsort_list.append(
            StrongSORT(
                strong_sort_weights,
                device,
                max_dist=cfg.STRONGSORT.MAX_DIST,
                max_iou_distance=cfg.STRONGSORT.MAX_IOU_DISTANCE,
                max_age=cfg.STRONGSORT.MAX_AGE,
                n_init=cfg.STRONGSORT.N_INIT,
                nn_budget=cfg.STRONGSORT.NN_BUDGET,
                mc_lambda=cfg.STRONGSORT.MC_LAMBDA,
                ema_alpha=cfg.STRONGSORT.EMA_ALPHA,

            )
        )
    outputs = [None] * nr_sources

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    t0 = time.time()

    # Run tracking
    # model.warmup(imgsz=(1 if pt else nr_sources, 3, *imgsz))  # warmup
    dt, seen = [0.0, 0.0, 0.0, 0.0], 0
    zone_class_cam1, giatri_count_cam1, name_class_cam1, zone_class_cam2, giatri_count_cam2, name_class_cam2 = 0,0,0,0,0,0
    curr_frames, prev_frames = [None] * nr_sources, [None] * nr_sources
    # for path, img, im0s, vid_cap in dataset:
    time_start = time.time()
    time_end = time.time()
    for frame_idx, (path, img, im0s, vid_cap) in enumerate(dataset):
        print("time per frame: ", time.time() - time_start)
        time_start = time.time()
        t1 = time_synchronized()
        # plt.imshow(np.transpose(img[0],(1,2,0)),interpolation='nearest')
        # cv2.imshow("raw image",img)
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        t2 = time_synchronized()
        dt[0] += t2 - t1
        # print("-------------img ------------")
        # print(img)
        # Inference
        pred = model(img, augment=opt.augment)[0]
        # print("-------------pred------------")
        # print(pred)
        t3 = time_synchronized()
        dt[1] += t3 - t2
        # Apply NMS
        pred = non_max_suppression(pred, 0.25, 0.45, classes=opt.classes, agnostic=opt.agnostic_nms)
        dt[2] += time_synchronized() - t3

        
        # Process detections
        for i, det in enumerate(pred):  # detections per image
            if webcam:  # batch_size >= 1
                p, s, im0, frame = path[i], '%g: ' % i, im0s[i].copy(), dataset.count

            else:
                p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)
            

            curr_frames[i] = im0
            p = Path(p)  # to Path
            txt_file_name = p.name
            
            s += '%gx%g ' % img.shape[2:]  # print string

            
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string
                    
                    if names[int(c)] == 'person' or names[int(c)] == 'worker':
                        num_person = n.item()               #so nguoi de xuat len gui
                
                
                xywhs = xyxy2xywh(det[:, 0:4])
                confs = det[:, 4]
                clss = det[:, 5]

                # pass detections to strongsort
                t4 = time_synchronized()
                outputs[i] = strongsort_list[i].update(xywhs.cpu(), confs.cpu(), clss.cpu(), im0)
                print(outputs[i])
                t5 = time_synchronized()
                dt[3] += t5 - t4
                # draw boxes for visualization
                if len(outputs[i]) > 0:

                    for j, (output, conf) in enumerate(zip(outputs[i], confs)):
                        bboxes = output[0:4]
                        id = output[4]
                        cls = output[5]
                        a_stt_zone = 0
                        bbox_left, bbox_top, bbox_right, bbox_bottom = bboxes
                        # detect_zone_list = main_win.detect_in_zone_yolo()
                        # for zone,points in detect_zone_list:
                        #     im0, stt_zone = define_zone(im0,output,zone, points)
                        #     if stt_zone != 0:
                        #         a_stt_zone = stt_zone
                        if save_txt:
                            # to MOT format
                            bbox_left = output[0]
                            bbox_top = output[1]
                            bbox_w = output[2] - output[0]
                            bbox_h = output[3] - output[1]
                            # Write MOT compliant results to file
                            if i == 0:
                                with open('dataframe.txt', 'a') as f:
                                    print("-------------------------------")
                                    f.write(('%g ' * 12 + '\n') % (frame_idx + 1, cls, id, bbox_left,  # MOT format, stt_zone: So thu tu zone
                                                                bbox_top, bbox_w, bbox_h,a_stt_zone, -1, -1, -1, -1))
                            if i == 1:
                                with open('dataframecam2.txt', 'a') as f:
                                    f.write(('%g ' * 12 + '\n') % (frame_idx + 1, cls, id, bbox_left,  # MOT format, stt_zone: So thu tu zone
                                                                bbox_top, bbox_w, bbox_h,a_stt_zone, -1, -1, -1, -1))
                        if True:  # Add bbox to image
                            c = int(cls)  # integer class
                            id = int(id)  # integer id
                            label = f'{id} {names[c]}'
                            plot_one_box(bboxes, im0, label=label, color=colors[int(cls)], line_thickness=line_thickness)
                            

                ### Print time (inference + NMS)
                print(f'{s}Done. YOLO:({t3 - t2:.3f}s), StrongSORT:({t5 - t4:.3f}s)')

            else:
                strongsort_list[i].increment_ages()
                print('No detections')


            if count:
                itemDict={}
                ## NOTE: this works only if save-txt is true
                try:
                    if i == 0:
                        df = pd.read_csv('dataframe.txt' , header=None, delim_whitespace=True)
                    if i == 1:
                        df = pd.read_csv('dataframecam2.txt' , header=None, delim_whitespace=True)
                    df = df.iloc[:,0:8].drop([3,4,5,6], axis = 1)               #giu lai frame_id, class, id_object, stt_zone
                    df.columns=["frameid" ,"class","trackid","zone"]
                    df = df[['class','trackid','zone']]
                    df = (df.groupby(by = ['zone','trackid'])['class']
                              .apply(list)                                          
                              .apply(lambda x:sorted(x))                            # group theo zone va trackid cua tung class
                             ).reset_index()
                    
                    df.colums = ["zone","trackid","class"]
                    df['class']=df['class'].apply(lambda x: Counter(x).most_common(1)[0][0]) # xoa mang tai cot class [0] -> 0

                    vc = df[['zone','class']].value_counts().sort_index()           #dem theo zone va trackid cua tung class
                    vc = dict(vc)
                    print(vc)
                    # vc1 = list(vc.keys())
                    vc2 = {}
                    for key, val in enumerate(names):
                        vc2[key] = val           # theo class_name trong tap train           
                    # itemDict = dict((vc2[key], value) for key, value in list(vc.items()))
                    
                    # itemDict  = dict(sorted(itemDict.items(), key=lambda item: item[0]))
                    itemDict = list(vc.items())   # {(zone, class): count}
                except:
                    pass

                if save_txt:
                    ## overlay
                    list_string_zone = []
                    list_report = []
                    for zone_class, giatri_count in itemDict: 
                        if zone_class[0] == 0:
                            continue
                        name_class = vc2[zone_class[1]]     # doi tu class digit sang class_name
                        if i == 0:
                            zone_class_cam1, giatri_count_cam1, name_class_cam1 = zone_class[0], giatri_count, name_class
                        if i ==1:
                            zone_class_cam2, giatri_count_cam2, name_class_cam2 = zone_class[0], giatri_count, name_class
                        if zone_class_cam1 == zone_class_cam2 and name_class_cam1 == name_class_cam2:
                            max_count = max(giatri_count_cam1,giatri_count_cam2)
                            string_zone = 'Khu vuc {} has {} {}'.format(zone_class[0], max_count, name_class)
                            
                        else: string_zone = 'Khu vuc {} has {} {}'.format(zone_class[0], giatri_count, name_class)
                        list_string_zone.append(string_zone)
                        list_report.append([zone_class[0],giatri_count, name_class])
                    # print(im0)
                    display = im0.copy()
                    h, w = im0.shape[0], im0.shape[1]
                    x1,y1,x2,y2 = 0,0,10,30
                    for string_zones in list_string_zone:
                        txt_size = cv2.getTextSize(str(string_zones), cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)[0]
                        cv2.rectangle(im0, (x1, y1 + 1), (txt_size[0] * 2, y2),(0, 0, 0),-1)
                        cv2.putText(im0, '{}'.format(string_zones), (x1 + 10, y1 + 20), cv2.FONT_HERSHEY_SIMPLEX,0.7, (210, 210, 210), 2)
                        cv2.addWeighted(im0, 0.7, display, 1 - 0.7, 0, im0)
                        y1 += 10
                        y2 += 30
                with open('dataframe.txt','w') as file:    ##### Xoa file txt de khong luu lai gia tri count, neu la video thi lui` lai 1 tab`
                    pass
                with open('dataframecam2.txt','w') as file:    ##### Xoa file txt de khong luu lai gia tri count, neu la video thi lui` lai 1 tab`
                    pass
            #current frame // tesing
            # cv2.imwrite('testing.jpg',im0)

            # Stream results
            if show_vid:
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)  # 1 millisecond


    if True:
        print(f"Results saved to ...")
    print(f'Done. ({time.time() - t0:.3f}s)')
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--line-thickness', default=1, type=int, help='bounding box thickness (pixels)')
    opt = parser.parse_args()
    #check_requirements(exclude=('pycocotools', 'thop'))

    with torch.no_grad():
        detect()