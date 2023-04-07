import argparse
import time
from pathlib import Path
import sys
import os

import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random

from yolov7.models.experimental import attempt_load
from yolov7.utils.datasets import LoadStreams, LoadImages
from yolov7.utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, \
    apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from yolov7.utils.plots import plot_one_box
from yolov7.utils.torch_utils import select_device, load_classifier, time_synchronized, TracedModel

from tracker.tracking import BoTSORT
# from tracker.mc_bot_sort import BoTSORT
from tracker.tracking_utils.timer import Timer
import easydict

import json
import math

sys.path.insert(0, './yolov7')
sys.path.append('.')

# source formats
img_formats = ['bmp', 'jpg', 'jpeg', 'png', 'tif', 'tiff', 'dng', 'webp', 'mpo']  # acceptable image suffixes
vid_formats = ['mov', 'avi', 'mp4', 'mpg', 'mpeg', 'm4v', 'wmv', 'mkv']  # acceptable video suffixes

def get_opt():
    '''
    parser = argparse.ArgumentParser()
    
    # parser.add_argument('--weights', nargs='+', type=str, default='yolov7.pt', help='model.pt path(s)') #
    # parser.add_argument('--ckpt-path', nargs='+', type=str, default='pretrained/yolov7.pt', help='model.pt path(s)')
    # parser.add_argument('--source', type=str, default='inference/images', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    print("hihddddi")
    parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold') #
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS') #
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', default=False, action='store_true', help='display results') #
    parser.add_argument('--save-txt', default=False, action='store_true', help='save results to *.txt') #
    parser.add_argument('--save-conf', default=False, action='store_true', help='save confidences in --save-txt labels') #
    parser.add_argument('--save-cmd', default=False, action='store_true', help='save command line in txt file') #
    parser.add_argument('--nosave', default=False, action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3') #
    parser.add_argument('--agnostic-nms', default=False, action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', default=False, action='store_true', help='augmented inference')
    parser.add_argument('--update', default=False, action='store_true', help='update all models')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name') #
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', default=False, action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--trace', default=False, action='store_true', help='trace model')
    parser.add_argument('--hide-labels-name', default=False, action='store_true', help='hide labels')

    # tracking args
    parser.add_argument("--track_high_thresh", type=float, default=0.3, help="tracking confidence threshold")
    parser.add_argument("--track_low_thresh", default=0.05, type=float, help="lowest detection threshold")
    parser.add_argument("--new_track_thresh", default=0.4, type=float, help="new track thresh")
    parser.add_argument("--track_buffer", type=int, default=30, help="the frames for keep lost tracks")
    parser.add_argument("--match_thresh", type=float, default=0.7, help="matching threshold for tracking")
    parser.add_argument("--aspect_ratio_thresh", type=float, default=1.6,
                        help="threshold for filtering out boxes of which aspect ratio are above the given value.")
    parser.add_argument('--min_box_area', type=float, default=10, help='filter out tiny boxes')
    parser.add_argument("--fuse-score", dest="mot20", default=False, action="store_true", #
                        help="fuse score and iou for association")

    # CMC
    parser.add_argument("--cmc-method", default="sparseOptFlow", type=str, help="cmc method: sparseOptFlow | files (Vidstab GMC) | orb | ecc")

    # ReID
    parser.add_argument("--with-reid", dest="with_reid", default=False, action="store_true", help="with ReID module.") #
    parser.add_argument("--fast-reid-config", dest="fast_reid_config", default=r"fast_reid/configs/MOT17/sbs_S50.yml",
                        type=str, help="reid config file path")
    parser.add_argument("--fast-reid-weights", dest="fast_reid_weights", default=r"pretrained/mot17_sbs_S50.pth",
                        type=str, help="reid config file path")
    parser.add_argument('--proximity_thresh', type=float, default=0.5,
                        help='threshold for rejecting low overlap reid matches')
    parser.add_argument('--appearance_thresh', type=float, default=0.25,
                        help='threshold for rejecting low appearance similarity reid matches')

    # Object detection
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
    
    # Keypoint extraction
    parser.add_argument('--kpt-thres', type=float, default=0.5, help='threshold for rejecting low confidence keypoints')
    parser.add_argument('--kpt-label', default=False, action='store_true', help='use keypoint labels')
    parser.add_argument('--nobbox', default=False, action='store_true', help='do not show bbox around person')

    # run mode
    parser.add_argument('--mode', type=str, default=None, help='run mode of model', choices=['detection', 'tracking'])
    '''

    opt = easydict.EasyDict({
        "img_size": 640,
        "conf_thres" : 0.25,
        "iou_thres" : 0.45,
        "device" : 'cpu',
        "view_img" : False,
        "save_txt" : False,
        "save_conf" :False,
        "save_cmd" : False,
        "nosave": False,
        # "classes": False,
        "agnostic_nms": False,
        "augment": False,
        "update": False,
        "project": 'runs/detect',
        "name": 'exp',
        "exist_ok": False,
        "trace": False,
        "hide_labels_name": False,
        "cmc_method": 'sparseOptFlow',
        "hide_labels_name": False,
        "with_reid": 'with_reid',
        "fast_reid_config": "fast_reid/configs/MOT17/sbs_S50.yml",
        "fast_reid_weights": "pretrained/mot17_sbs_S50.pth",
        "proximity_thresh": 0.5,
        "appearance_thresh": 0.25,
        "hide_labels": False,
        "hide_conf": False,
        "kpt_thres": 0.5,
        "kpt_label": False,
        "nobbox": False,
        "mode": None,
        "jde": False,
        "ablation": False,
        "track_high_thresh": 0.3,
        "track_low_thresh": 0.05,
        "new_track_thresh": 0.4,
        "track_buffer": 30,
        "match_thresh": 0.7,
        "aspect_ratio_thresh": 1.6,
        "min_box_area": 10,
        "fuse-score": False,
        "mot20": False
        # "ablation": False,
        # "ablation": False,
        # "ablation": False,
        # "ablation": False,
    })
    
    #opt = parser.parse_args()

    # opt.jde = False
    # opt.ablation = False

    print(opt)

    return opt

def detect(mode, data_path, save_img=False):
    """
    Descripiton:
    Args:
    Return:
    """
    ### Assertion
    assert mode in ['detection', 'tracking'], f'ERROR: Invalid mode {mode} designated'
    assert os.path.exists(data_path), f'ERROR: Invalid data path {data_path} designated'
    
    ### Parsing arguments
    opt = get_opt()
    run_mode = mode
    source = data_path
    weights = 'pretrained/yolov7.pt' if mode == 'detection' else 'pretrained/yolov7-w6-pose.pt'
    src_format = source.split('.')[-1]


    first_frame_img_path = '../runs/image.jpg'
    # detection mode
    if run_mode == 'detection':
        # it should be image or video
        assert src_format in (img_formats + vid_formats), 'ERROR: Invalid source format'
        # assert model weight
        assert weights.split('/')[-1] == 'yolov7.pt', 'ERROR: You must use "yolov7.pt" weight file'
        # dictionary for returning json file
        det_result = {}

    # tracking mode
    elif run_mode == 'tracking':
        # it must be a video
        assert src_format in vid_formats, 'ERROR: Source must be a video in tracking mode'
        # assert model weight
        assert weights.split('/')[-1] == 'yolov7-w6-pose.pt', 'ERROR: You must use "yolov7-w6-pose.pt" weight file'
        # dictionary for returning json file
        kpt_result, mot_result = {}, {}

    # flag: display demo, flag: save result in txt, flag: save command in txt, image size for inference (resize), flag: trace model, flag: save keypoint labels
    view_img, save_json, save_txt, save_cmd, imgsz, trace, kpt_label = (
        opt.view_img, True, False, False, opt.img_size, opt.trace, (run_mode == 'tracking')
    )
    # flag: model hparams
    conf_thres, iou_thres, save_conf, classes, fuse_score, agnostic_nms = (
        opt.conf_thres, opt.iou_thres, True, [0], True, True
    )
    # flag: save result in video (or images)
    # save_img = not opt.nosave and not source.endswith('.txt')
    # webcam
    webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
        ('rtsp://', 'rtmp://', 'http://', 'https://'))

    ### Making directories (No saving)
    # directory path for saving experiment (increment run)
    save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))
    save_dir.mkdir(parents=True, exist_ok=True)

    ### Initialize
    # set logging format
    set_logging()
    # cuda device or cpu
    if torch.cuda.device_count() > 0 and torch.cuda.is_available():
        opt.device = '0'
    else:
        opt.device = 'cpu'
    device = select_device(opt.device)
    # half precision only supported on CUDA
    print(device.type)
    half = (device.type != 'cpu')

    ### Load model
    # Load YOLOv7 model with designated weight path (FP32 model)
    model = attempt_load(weights, map_location=device)
    print(f"Load weights: {weights}")
    # model stride
    stride = int(model.stride.max())
    # inference size
    imgsz = check_img_size(imgsz, s=stride)
    # save model weight (trace version)
    if trace: model = TracedModel(model, device, imgsz)
    # to FP16
    if half:  model.half()  

    ### Second-stage classifier
    classify = False
    if classify:
        modelc = load_classifier(name='resnet101', n=2)  # initialize
        modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model']).to(device).eval()

    ### Set Dataloader
    vid_path, vid_writer = None, None
    if webcam:
        view_img = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride)

    ### Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(100)]

    ### Create tracker
    fps = int(cv2.VideoCapture(source).get(cv2.CAP_PROP_FPS))
    if run_mode == 'tracking':
        opt.with_reid = (opt.device != 'cpu') 
        # thresh: the lower, the robuster (distance)
        opt.proximity_thresh = 0.5 # 0.5
        opt.appearance_thresh = 0.25 # 0.25
        opt.match_thresh = 1.0 # 0.7
        tracker = BoTSORT(opt, frame_rate=fps)

    ### Run inference
    # On gpu
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    # check inference time
    t0 = time.time()
    # for each frame, 
    n = 1 # every 1 second
    sec = -n
    for frame_id, (path, img, im0s, vid_cap) in enumerate(dataset):
        # Skip non-checking frames
        if frame_id % int(fps * n) != 0:
            tracker.frame_id += 1
            continue
        # Increase current second
        sec += n
        # to gpu
        img = torch.from_numpy(img).to(device)
        # uint8 to fp16/32
        img = img.half() if half else img.float()
        # normalize (0 - 255 to 0.0 - 1.0)
        img /= 255.0
        # (ch, w, h) -> (1, ch, w, h)
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        ## Inference
        # detection using YOLOv7
        pred = model(img, augment=opt.augment)[0]

        ## Apply NMS
        # detection result after NMS
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes=classes, agnostic=agnostic_nms, kpt_label = kpt_label)

        ## Apply Classifier
        if classify:
            pred = apply_classifier(pred, modelc, img, im0s)

        ## Process detections
        results = []
        # For detections of current frame, 
        for i, det in enumerate(pred):
            # path, string, origin frame, frame index
            if webcam:  # batch_size >= 1
                p, s, im0, frame = path[i], '%g: ' % i, im0s[i].copy(), dataset.count
            else:
                p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)

            # Normalization gain whwh (for saving result)
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]
            # Run tracker
            detections = []

            # NOTE) Obtain adjusted coordinates of detection bboxes and keypoints
            if len(det):
                # scaled coordinates wrt origin frame
                boxes = scale_coords(img.shape[2:], det[:, :4], im0.shape, kpt_label=False)
                if run_mode == 'tracking':
                    kpt_boxes = scale_coords(img.shape[2:], det[:, 6:], im0.shape, kpt_label=kpt_label, step=3)
                
                # to cpu
                boxes = boxes.cpu().numpy()
                if run_mode == 'tracking':
                    kpt_boxes = kpt_boxes.cpu().numpy()
                detections = det.cpu().numpy()
                # replace to scaled coordinaets of bbox
                detections[:, :4] = boxes
                # replace to scaled coordinates of kpts
                if run_mode == 'tracking':
                    detections[:, 6:] = kpt_boxes

                # Save results of detection
                if run_mode == 'detection':
                    # For each detection, 
                    for det_index, (*xyxy, conf, cls) in enumerate(reversed(detections[:, :6])):
                        # normalized xywh
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist() 
                        # label format
                        x1, y1, x2, y2 = detections[det_index, :4]
                        w, h = x2 - x1, y2 - y1
                        s = conf
                        # Save detection result
                        if save_json:
                            det_result[det_index + 1] = {
                                    "x1": round(float(x1), 4), 
                                    "y1": round(float(y1), 4), 
                                    "w": round(float(w), 4), 
                                    "h": round(float(h), 4), 
                                    "s": round(float(s), 4) 
                            }
                        # Write to file
                        if save_txt: 
                            det_line = (det_index + 1, *list(map(lambda x: round(x, 4), [x1, y1, w, h, s])))
                            # save detection result
                            with open(det_path, 'a') as f:
                                f.write(('%g ' * len(det_line)).rstrip() % det_line+ '\n')
                        # Draw bbox to an image
                        if save_img: 
                            c = int(cls)
                            if opt.hide_labels_name:
                                label = None
                            else:
                                label = (names[c] if opt.hide_conf else f'{names[c]} {conf:.2f}')
                            plot_one_box(xyxy, im0, 
                                        label=label, 
                                        color=colors[c], 
                                        line_thickness=2,
                                        orig_shape=im0.shape[:2],
                                        nobbox=False)
                    # Return detection result
                    # p = Path(p)  # to Path
                    # save_path = str(save_dir / p.name).split('.')[0] + '.jpg'  # img.jpg
                    # cv2.imwrite(save_path, im0)
                    print(f'Detection done. ({time.time() - t0:.3f}s)')
                    return im0, det_result
                    # Save json file
                    # p = Path(p)  # to Path
                    # save_path = str(save_dir / 'det_result.json')
                    # if save_json:
                    #     with open(save_path, 'w') as f:
                    #         json.dump(det_result, f, indent=4)
                    # Save result image (image with detections)
                    # save_path = str(save_dir / p.name).split('.')[0] + '.jpg'  # img.jpg
                    # if save_img:
                    #     cv2.imwrite(save_path, im0)
            else:
                if mode == 'detection': return det_result
                else: pass
                
            # args: scaled coordinates wrt origin frame, origin frame image
            # online trackers
            online_targets = tracker.update(detections, im0)
            # online (t, l, w, h)
            online_tlwhs = []
            # online object ID
            online_ids = []
            # online object score
            online_scores = []
            # online object class
            online_cls = []
            # online object keypoints per frame
            online_kpts = []

            # for each tracking
            kpt_result[sec] = {}
            mot_result[sec] = {}
            for ti, t in enumerate(online_targets):
                tlwh = t.tlwh
                tlbr = t.tlbr
                xywh = t.xywh
                tid = t.track_id
                tcls = t.cls
                tkpts = t.kpts
                kpt_result[sec][tid] = {}
                # if area of bbox is bigger than min_box_area,
                if tlwh[2] * tlwh[3] > opt.min_box_area:
                    # [t, l, w, h, id, score, class] for each online valid bboxes
                    online_tlwhs.append(tlwh)
                    online_ids.append(tid)
                    online_scores.append(t.score)
                    online_cls.append(t.cls)

                    # save tracking results
                    # frame i, tracking id, x, y, w, h, confidence, -1, -1, -1
                    results.append(
                        f"{i + 1},{tid},{tlwh[0]:.2f},{tlwh[1]:.2f},{tlwh[2]:.2f},{tlwh[3]:.2f},{t.score:.2f},-1,-1,-1\n"
                    )

                    # save mot result of current frame
                    (x1, y1, w, h), s = tlwh[:4], t.score
                    if save_json:
                        mot_result[sec][tid] = {
                            "x1": round(float(x1), 4), 
                            "y1": round(float(y1), 4), 
                            "w": round(float(w), 4), 
                            "h": round(float(h), 4), 
                            "s": round(float(s), 4) 
                        }
                    if save_txt:
                        mot_line = (sec, tid, *list(map(lambda x: round(x, 4), [x1, y1, w, h, s])))
                        with open(mot_path, 'a') as f:
                            f.write(('%g ' * len(mot_line)).rstrip() % mot_line + '\n')

                    # save keypoint extraction result of current frame
                    # tkpts: dict = {0: [t, l, b, r, s, c, x0, y0, s1, x1, y1, s1, ...], 1: [...], ...}
                    if frame_id in tkpts:
                        tkpt = tkpts[frame_id]
                    else:
                        continue
                    k_step = 3
                    num_kpts = len(tkpt) // k_step
                    for kid in range(num_kpts):
                        x_coord, y_coord = tkpt[k_step * kid], tkpt[k_step * kid + 1]
                        if not (x_coord % 640 == 0 or y_coord % 640 == 0):
                            if save_json:
                                conf = tkpt[k_step * kid + 2]
                                x, y, s = x_coord, y_coord, conf
                                kpt_result[sec][tid][kid] = {
                                    "x": round(float(x), 4), 
                                    "y": round(float(y), 4), 
                                    "s": round(float(s), 4)
                                }
                            if save_txt:
                                kpt_line = (sec, tid, *list(map(lambda x: round(x, 4), [x, y, s])), kid)
                                with open(kpt_path, 'a') as f:
                                    f.write(('%g ' * len(kpt_line)).rstrip() % kpt_line + '\n')
                        
                    if save_txt:
                        with open(kpt_path, 'a') as f:
                            f.write('\n')

                    # Add bbox and kpts to image
                    if save_img or view_img: 
                        if opt.hide_labels_name:
                            label = f'{tid}, {int(tcls)}'
                        else:
                            label = f'{tid}, {names[int(tcls)]}'
                        plot_one_box(tlbr, im0, 
                                     label=label, 
                                     color=colors[int(tid) % len(colors)], 
                                     line_thickness=2, 
                                     kpt_label=kpt_label, 
                                     kpts=tkpt, 
                                     steps=k_step, 
                                     orig_shape=im0.shape[:2],
                                     nobbox=False)
                                     
            # p = Path(p)  # to Path
            # save_path = str(save_dir / p.name)  # img.jpg

            # Stream results
            if view_img:
                cv2.imshow('BoT-SORT', im0)
                cv2.waitKey(1)  # 1 millisecond

            # Save results (image with tracking + keypoints)
            if save_img:
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)
                else:  # 'video' or 'stream'
                    if vid_path != save_path:  # new video
                        vid_path = save_path
                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release()  # release previous video writer
                        if vid_cap:  # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # stream
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                            save_path += '.mp4'
                        vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    vid_writer.write(im0)

    # Return keypoint extraction and tracking result
    print(f'Tracking done. ({time.time() - t0:.3f}s)')
    return kpt_result, mot_result
    # save json file
    # save_path = str(save_dir / 'kpt_result.json')
    # with open(save_path, 'w') as f:
    #     json.dump(kpt_result, f, indent=4)
    # save_path = str(save_dir / 'mot_result.json')
    # with open(save_path, 'w') as f:
    #     json.dump(mot_result, f, indent=4)


def run_model(mode, data_path):
    with torch.no_grad():
        return detect(mode, data_path)
