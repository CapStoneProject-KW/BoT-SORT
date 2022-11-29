import argparse
import time
from pathlib import Path
import sys

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

sys.path.insert(0, './yolov7')
sys.path.append('.')

def write_results(filename, results):
    title = f'frame\tid\tx\ty\tw\th\ts\n'
    with open(filename, 'w') as f:
        f.write(title)
        f.write(''.join(results))
    print('Saved results to {}'.format(filename))


def detect(save_img=False):
    ### Parsing arguments
    # video path, model weight path, flag: display demo, flag: save result in txt, flag: save command in txt, image size for inference (resize), flag: trace model
    source, weights, view_img, save_txt, save_cmd, imgsz, trace = opt.source, opt.weights, opt.view_img, opt.save_txt, opt.save_cmd, opt.img_size, opt.trace
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
    # model stride
    stride = int(model.stride.max())
    # inference size
    imgsz = check_img_size(imgsz, s=stride)

    # save model weight
    if trace: model = TracedModel(model, device, opt.img_size)
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
    tracker = BoTSORT(opt, frame_rate=30.0)

    ### Run inference
    # On gpu
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    # check inference time
    t0 = time.time()
    # results of MOT
    # format: frame_id, track_id, x, y, w, h, s
    mot_result = []
    # for each frame, 
    # video path, converted image, origin frame, video capture object
    for frame_id, (path, img, im0s, vid_cap) in enumerate(dataset):
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
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
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
        for i, det in enumerate(pred):
            # path, string, origin frame, frame index
            if webcam:  # batch_size >= 1
                p, s, im0, frame = path[i], '%g: ' % i, im0s[i].copy(), dataset.count
            else:
                p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)

            # Run tracker
            detections = []
            if len(det):
                # scaled coordinates wrt origin frame
                boxes = scale_coords(img.shape[2:], det[:, :4], im0.shape)
                # to cpu
                boxes = boxes.cpu().numpy()
                detections = det.cpu().numpy()
                # replace to scaled coordinaets of bbox
                detections[:, :4] = boxes

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

            # for each tracking
            for t in online_targets:
                tlwh = t.tlwh
                tlbr = t.tlbr
                tid = t.track_id
                tcls = t.cls
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
                    # NOTE) Consider saving coordinates from 'detections' list
                    mot_result.append(
                        f"{frame_id}\t{tid}\t{tlwh[0]:5.2f}\t{tlwh[1]:5.2f}\t{tlwh[2]:5.2f}\t{tlwh[3]:5.2f}\t{t.score:5.2f}\n"
                    )

                    if save_img or view_img:  # Add bbox to image
                        if opt.hide_labels_name:
                            label = f'{tid}, {int(tcls)}'
                        else:
                            label = f'{tid}, {names[int(tcls)]}'
                        plot_one_box(tlbr, im0, label=label, color=colors[int(tid) % len(colors)], line_thickness=2)
            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # img.jpg

            # Print time (inference + NMS)
            # print(f'{s}Done. ({t2 - t1:.3f}s)')

            # Stream results
            if view_img:
                cv2.imshow('BoT-SORT', im0)
                cv2.waitKey(1)  # 1 millisecond

            # Save results (image with detections)
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

    ### Save MOT result in txt
    if save_txt or save_img:
        # print(len(results))
        write_results(f'{save_dir}/labels.txt', mot_result)
        # s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        # print(f"Results saved to {save_dir}{s}")

    print(f'Done. ({time.time() - t0:.3f}s)')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='yolov7.pt', help='model.pt path(s)') #
    parser.add_argument('--source', type=str, default='inference/images', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--img-size', type=int, default=1920, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.09, help='object confidence threshold') #
    parser.add_argument('--iou-thres', type=float, default=0.7, help='IOU threshold for NMS') #
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
