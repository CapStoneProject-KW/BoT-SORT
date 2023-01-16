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

from tracker.mc_bot_sort import BoTSORT
from tracker.tracking_utils.timer import Timer

import json
import math

sys.path.insert(0, './yolov7')
sys.path.append('.')

# source formats
img_formats = ['bmp', 'jpg', 'jpeg', 'png', 'tif', 'tiff', 'dng', 'webp', 'mpo']  # acceptable image suffixes
vid_formats = ['mov', 'avi', 'mp4', 'mpg', 'mpeg', 'm4v', 'wmv', 'mkv']  # acceptable video suffixes


def write_results(filename, results):
    title = f'frame\tid\tx\ty\tw\th\ts\n'
    with open(filename, 'w') as f:
        f.write(title)
        f.write(''.join(results))
    print('Saved results to {}'.format(filename))

def detect(save_img=False):
    ### Parsing arguments
    run_mode = opt.mode
    source = opt.source
    src_format = source.split('.')[-1]
    # detection mode
    if run_mode == 'detection':
        # it should be image or video
        assert src_format in (img_formats + vid_formats), 'ERROR: Invalid source format'
        # designate model weight
        weights = 'pretrained/yolov7.pt'
    # tracking mode
    elif run_mode == 'tracking':
        # it must be a video
        assert src_format in vid_formats, 'ERROR: Source must be a video in tracking mode'
        # designate model weight
        weights = 'pretrained/yolov7-w6-pose.pt'
    # invalid mode
    else:
        print(f"ERROR: Invalid '{run_mode}' mode designated")
        exit(-1)

    # flag: display demo, flag: save result in txt, flag: save command in txt, image size for inference (resize), flag: trace model, flag: save keypoint labels
    view_img, save_json, save_txt, save_cmd, imgsz, trace, kpt_label = (
        opt.view_img, True, True, True, opt.img_size, opt.trace, (run_mode == 'tracking')
    )
    # 
    conf_thres, iou_thres, save_conf, classes, fuse_score, agnostic_nms, with_reid = (
        opt.conf_thres, opt.iou_thres, True, [0], True, True, True
    )
    '''
    # video path, model weight path, flag: display demo, flag: save result in txt, flag: save command in txt, image size for inference (resize), flag: trace model
    source, weights, view_img, save_txt, save_cmd, imgsz, trace = opt.source, opt.weights, opt.view_img, opt.save_txt, opt.save_cmd, opt.img_size, opt.trace
    kpt_label, det_mode, kpt_mode = opt.kpt_label, opt.det_mode, opt.kpt_mode
    '''
    # flag: save result in video (or images)
    save_img = not opt.nosave and not source.endswith('.txt')
    # webcam
    webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
        ('rtsp://', 'rtmp://', 'http://', 'https://'))

    ### Making directories
    # directory path for saving experiment (increment run)
    save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))
    save_dir.mkdir(parents=True, exist_ok=True)
    # (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir
    # save command in txt
    if save_cmd:
        with open(f'{save_dir}/cmd.txt', 'w') as f:
            f.write('python' + ' ' + ' '.join(sys.argv))

    if run_mode == 'detection':
        det_path = str(save_dir / 'det_result.txt')
        with open(det_path, 'w') as f:
            f.write('Labels: (idx x1 y1 w h s)\n')
    elif run_mode == 'tracking':
        mot_path = str(save_dir / 'mot_result.txt')
        kpt_path = str(save_dir / 'kpt_result.txt')
        with open(mot_path, 'w') as f:
            f.write('Labels: (fid tid x1 y1 w h s)\n')
        with open(kpt_path, 'w') as f:
            f.write('Labels: (fid tid x y s kid)\n')

    ### Initialize
    # set logging format
    set_logging()
    # cuda device or cpu
    device = select_device(opt.device)
    # half precision only supported on CUDA
    half = device.type != 'cpu'

    ### Load model
    # Load YOLOv7 model with designated weight path (FP32 model)
    model = attempt_load(weights, map_location=device)
    print(f"Load weights: {weights}")
    # print(model)
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
    if run_mode == 'tracking':
        tracker = BoTSORT(opt, frame_rate=30.0)

    ### Run inference
    # On gpu
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    # check inference time
    t0 = time.time()
    # Keypoint and MOT result
    kpt_result = {}
    mot_result = {}
    # for each frame, 
    # video path, converted image, origin frame, video capture object
    for frame_id, (path, img, im0s, vid_cap) in enumerate(dataset):
        # Process for every n sec
        '''
        For 22 sec/24 fps video, 
        n = 0.1: 40~41 sec
        n = 0.2: 20~21 sec
        n = 0.5: 7~8 sec
        n = 1.0: 3~4 sec
        '''
        n = 1.0
        fps = int(vid_cap.get(cv2.CAP_PROP_FPS))
        if frame_id % int(fps * n) != 0:
            tracker.frame_id += 1
            continue
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
        # pytorch-accurate time
        # t1 = time_synchronized()
        # detection using YOLOv7
        pred = model(img, augment=opt.augment)[0]

        ## Apply NMS
        # detection result after NMS
        '''Example
        # x, y, w, h, conf, class
        [tensor([[ 91.31250, 242.00000, 204.25000, 549.00000,   0.92432,   0.00000],
        [184.00000, 241.00000, 273.50000, 557.00000,   0.91113,   0.00000]], device='cuda:0')]
        '''
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes=classes, agnostic=agnostic_nms, kpt_label = kpt_label)
        # pytorch-accurate time
        # t2 = time_synchronized()

        ## Apply Classifier
        if classify:
            pred = apply_classifier(pred, modelc, img, im0s)

        ## Process detections
        ''' Example
        frame		id		x		    y		    w		    h		    s
        1		    1		2.47		241.07		278.74		479.62		0.87
        1		    2		183.24		187.96		223.91		530.23		0.89
        '''
        results = []
        # For detections of current frame, 
        '''det
        0~3: xyxy
        4: conf
        5: class
        6~: 17 keypoints (x, y, conf)
        '''
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
                # detections[:, :4] = boxes
                boxes = scale_coords(img.shape[2:], det[:, :4], im0.shape, kpt_label=False)
                # det[:, 6:] == kpt_boxes
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
                    det_result = {}
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
                                    "x1": float(round(x1, 2)), 
                                    "y1": float(round(y1, 2)), 
                                    "w": float(round(w, 2)), 
                                    "h": float(round(h, 2)), 
                                    "s": float(round(s, 2)) 
                            }
                        # Write to file
                        if save_txt: 
                            det_line = (frame_id, det_index + 1, *list(map(lambda x: round(x, 2), [x1, y1, w, h, s])))
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
                    # Save result image (image with detections)
                    p = Path(p)  # to Path
                    save_path = str(save_dir / 'det_result.json')
                    if save_json:
                        with open(save_path, 'w') as f:
                            json.dump(det_result, f, indent=4)
                    save_path = str(save_dir / p.name).split('.')[0] + '.jpg'  # img.jpg
                    if save_img:
                        cv2.imwrite(save_path, im0)

                    print(f'Detection done. ({time.time() - t0:.3f}s)')
                    return
                
            # NOTE) Process trackers 
            '''Example
            (for tenth frame)
            OT_1_(1-10)
            OT_2_(1-10)
            ==================
            repr: 'OT_{}_({}-{})'.format(self.track_id, self.start_frame, self.end_frame)
            '''
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
            kpt_result[frame_id] = {}
            mot_result[frame_id] = {}
            for ti, t in enumerate(online_targets):
                tlwh = t.tlwh
                tlbr = t.tlbr
                xywh = t.xywh
                tid = t.track_id
                tcls = t.cls
                tkpts = t.kpts
                kpt_result[frame_id][tid] = {}
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
                    mot_line = (frame_id, tid, *list(map(lambda x: round(x, 2), [x1, y1, w, h, s])))
                    if run_mode == 'tracking':
                        mot_result[frame_id][tid] = {
                            "x1": float(round(x1, 2)), 
                            "y1": float(round(y1, 2)), 
                            "w": float(round(w, 2)), 
                            "h": float(round(h, 2)), 
                            "s": float(round(s, 2)) 
                        }
                        with open(mot_path, 'a') as f:
                            f.write(('%g ' * len(mot_line)).rstrip() % mot_line + '\n')

                    # save keypoint extraction result of current frame
                    # tkpts: dict = {0: [t, l, b, r, s, c, x0, y0, s1, x1, y1, s1, ...], 1: [...], ...}
                    if frame_id in tkpts:
                        tkpt = tkpts[frame_id][6:]
                    else:
                        continue
                    k_step = 3
                    num_kpts = len(tkpt) // k_step
                    for kid in range(num_kpts):
                        x_coord, y_coord = tkpt[k_step * kid], tkpt[k_step * kid + 1]
                        if not (x_coord % 640 == 0 or y_coord % 640 == 0):
                        # if x_coord % imgsz != 0 and y_coord % imgsz != 0:
                            conf = tkpt[k_step * kid + 2]
                            # dont save the keypoint if its score is less than kpt_thres
                            '''
                            if conf < opt.kpt_thres:
                                continue
                            '''
                            x, y, s = x_coord, y_coord, conf
                            kpt_result[frame_id][tid][kid] = {
                                "x": float(round(x, 2)), 
                                "y": float(round(y, 2)), 
                                "s": float(round(s, 2))
                            }
                            kpt_line = (frame_id, tid, *list(map(lambda x: round(x, 2), [x, y, s])), kid)
                            if run_mode == 'tracking':
                                with open(kpt_path, 'a') as f:
                                    f.write(('%g ' * len(kpt_line)).rstrip() % kpt_line + '\n')
                        
                    if run_mode == 'tracking':
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
                                     
            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # img.jpg

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

    save_path = str(save_dir / 'kpt_result.json')
    with open(save_path, 'w') as f:
        json.dump(kpt_result, f, indent=4)
    save_path = str(save_dir / 'mot_result.json')
    with open(save_path, 'w') as f:
        json.dump(mot_result, f, indent=4)
    
    
    print(f'Tracking done. ({time.time() - t0:.3f}s)')
    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='yolov7.pt', help='model.pt path(s)') #
    parser.add_argument('--source', type=str, default='inference/images', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold') #
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS') #
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results') #
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt') #
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels') #
    parser.add_argument('--save-cmd', action='store_true', help='save command line in txt file') #
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3') #
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name') #
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--trace', action='store_true', help='trace model')
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
    parser.add_argument('--kpt-label', action='store_true', help='use keypoint labels')
    parser.add_argument('--nobbox', action='store_true', help='do not show bbox around person')

    # run mode
    parser.add_argument('--mode', type=str, default=None, help='run mode: detection or tracking')
    # parser.add_argument('--det-mode', action='store_true', help='only detection mode')
    # parser.add_argument('--kpt-mode', action='store_true', help='detection + keypoint mode')
    # parser.add_argument('--track-mode', action='store_true', help='full mode: detection + keypoint + tracking')

    
    opt = parser.parse_args()

    opt.jde = False
    opt.ablation = False

    print(opt)
    # check_requirements(exclude=('pycocotools', 'thop'))

    with torch.no_grad():
        if opt.update:  # update all models (to fix SourceChangeWarning)
            for opt.weights in ['yolov7.pt']:
                detect()
                strip_optimizer(opt.weights)
        else:
            detect()
