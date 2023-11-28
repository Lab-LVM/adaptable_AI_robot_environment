# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Run YOLOv5 detection inference on images, videos, directories, globs, YouTube, webcam, streams, etc.

Usage - sources:
    $ python detect.py --weights yolov5s.pt --source 0                               # webcam
                                                     img.jpg                         # image
                                                     vid.mp4                         # video
                                                     path/                           # directory
                                                     'path/*.jpg'                    # glob
                                                     'https://youtu.be/Zgi9g1ksQHc'  # YouTube
                                                     'rtsp://example.com/media.mp4'  # RTSP, RTMP, HTTP stream

Usage - formats:
    $ python detect.py --weights yolov5s.pt                 # PyTorch
                                 yolov5s.torchscript        # TorchScript
                                 yolov5s.onnx               # ONNX Runtime or OpenCV DNN with --dnn
                                 yolov5s.xml                # OpenVINO
                                 yolov5s.engine             # TensorRT
                                 yolov5s.mlmodel            # CoreML (macOS-only)
                                 yolov5s_saved_model        # TensorFlow SavedModel
                                 yolov5s.pb                 # TensorFlow GraphDef
                                 yolov5s.tflite             # TensorFlow Lite
                                 yolov5s_edgetpu.tflite     # TensorFlow Edge TPU
                                 yolov5s_paddle_model       # PaddlePaddle
"""

import argparse
import os
import platform
import sys
from pathlib import Path
import pathlib
import torch

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadStreams
from utils.general import (LOGGER, Profile, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_coords, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, smart_inference_mode

weights=ROOT / 'yolov5s.pt',  # model.pt path(s)
source=ROOT / 'data/images',  # file/dir/URL/glob, 0 for webcam
data=ROOT / 'data/coco128.yaml',  # dataset.yaml path
imgsz=(640, 480),  # inference size (height, width)
conf_thres=0.25,  # confidence threshold
unknownconf_thres=0.25,
iou_thres=0.45,  # NMS IOU threshold
max_det=1000,  # maximum detections per image
device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
view_img=False,  # show results
save_txt=False,  # save results to *.txt
save_conf=False,  # save confidences in --save-txt labels
save_crop=False,  # save cropped prediction boxes
nosave=False,  # do not save images/videos
classes=None,  # filter by class: --class 0, or --class 0 2 3
agnostic_nms=False,  # class-agnostic NMS
augment=False,  # augmented inference
visualize=False,  # visualize features
update=False,  # update all models
project=ROOT / 'runs/detect',  # save results to project/name
name='exp',  # save results to project/name
exist_ok=False,  # existing project/name ok, do not increment
line_thickness=3,  # bounding box thickness (pixels)  中文 * 2
hide_labels=False,  # hide labels
hide_conf=False,  # hide confidences
half=False,  # use FP16 half-precision inference
dnn=False,  # use OpenCV DNN for ONNX inference
vid_stride=1,  # video frame-rate stride


def convert(size, box):
    dw = 1./(size[0])
    dh = 1./(size[1])
    x = (box[0] + box[1])/2.0 - 1
    y = (box[2] + box[3])/2.0 - 1
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x*dw
    w = w*dw
    y = y*dh
    h = h*dh
    return (x,y,w,h)

def flask_detect(im, model, stride, pt, names, filename, unknown_start_index):

    yolo_format_list = []
    label_list = []
    
    source = str(im)
    save_img = not nosave and not source.endswith('.txt') 
    is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
    is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
    webcam = source.isnumeric() or source.endswith('.txt') or (is_url and not is_file)
    if is_url and is_file:
        source = check_file(source)  # download
    source = source.split('jpg/')[0]+'jpg'
    print('source : ', source)
    save_dir = './static/detection_save'

    if webcam:
        view_img = check_imshow()
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
        bs = len(dataset) 
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
        bs = 1 
    vid_path, vid_writer = [None] * bs, [None] * bs

    anno_save_dir = './annosave' 
    
    # Run inference
    model.warmup(imgsz=(1 if pt else bs, 3, *imgsz))  
    seen, windows, dt = 0, [], (Profile(), Profile(), Profile())
    for path, im, im0s, vid_cap, s in dataset:
        with dt[0]:
            im = torch.from_numpy(im)
            im = im.half() if model.fp16 else im.float() 
            im /= 255  
            if len(im.shape) == 3:
                im = im[None] 

        with dt[1]:
            pred = model(im, augment=augment, visualize=visualize)

 
        with dt[2]:
            pred = non_max_suppression(pred, conf_thres[0], iou_thres[0], classes[0], agnostic_nms, max_det=max_det[0], unknownconf=unknownconf_thres[0])

        for i, det in enumerate(pred):  
            seen += 1
            if webcam:  
                p, im0, frame = path[i], im0s[i].copy(), dataset.count
                s += f'{i}: '
            else:
                p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)

            p = Path(p)  
            save_path = './static/detection_save' + '/' + p.name
            txt_path = './static/detection_save/labels' + '/' + p.name + ('' if dataset.mode == 'image' else f'_{frame}')
            s += '%gx%g ' % im.shape[2:]  
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  
            imc = im0.copy() if save_crop else im0  
            annotator = Annotator(im0, line_width=line_thickness, example=str(names))
            if len(det):
                det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()
                for c in det[:, 5].unique():
                    n = (det[:, 5] == c).sum()  
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, " 

                for *xyxy, conf, cls in reversed(det):
                    if save_txt[0]: 
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  
                        line = (cls, *xywh, conf) if save_conf else (cls, *xywh) 

                        with open(f'{txt_path}.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')

                    if save_img or save_crop or view_img:  
                        c = int(cls)
     
                        hide_class = [] 
                        if c in hide_class:
          
                      
                        label = None if hide_labels[0] else (names[c] if hide_conf[0] else f'{names[c]} {conf:.2f}')
                        annotator.box_label(xyxy, label, color=colors(c, True))
                        voc_format = (float(xyxy[0]), float(xyxy[2]), float(xyxy[1]), float(xyxy[3]))
                        yolo_format = convert((im0.shape[1], im0.shape[0]), voc_format)
                        label_list.append(str(label.split(' ')[0]))
                        if c >= 80 : 
                            c = unknown_start_index
                        yolo_format_list.append(str(c) + " " + " ".join([str(a) for a in yolo_format]) + '\n')
   
                    if save_crop[0]:
                        save_one_box(xyxy, imc, file=save_dir / 'crops' / names[c] / f'{p.stem}.jpg', BGR=True)

            # Stream results
            im0 = annotator.result()
            cv2.imwrite(save_path, im0)
            
        LOGGER.info(f"{s}{'' if len(det) else '(no detections), '}{dt[1].dt * 1E3:.1f}ms")


    t = tuple(x.t / seen * 1E3 for x in dt)  

    if 'unknown' in label_list:
        out_file = open(anno_save_dir + '/' + filename + '.txt', 'w' ,encoding='UTF8')
        for i in yolo_format_list:
            out_file.write(i)
        print('annotation save done')

    LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}' % t)
    print('#'*100)

