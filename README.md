# MOT-YOLOv7-pose (full ver.)

## Introduction

_TBU_ 

## NOTE) How to use?

**Environment setup**

* Use `my_requirements.yaml` file
* NOTE) You need to install appropriate version of cuda(11.6), cudnn, and pytorch first
* NOTE) If any problem occurs, please check [here](https://github.com/NirAharon/BoT-SORT#installation) for further information

```shell
conda env create -f my_requirements.yaml
```

**Run model**

After you run the model, you can find the results at `{path_to_save_project}` directory 

* Detection mode (Only detection for the first frame of the video)

```shell
python tools/mc_demo_yolov7.py --mode detection --source {path_to_video} --project {path_to_save_project}
```

* Tracking mode (Keypoint extraction + Multiple object tracking for the entire length of the video)

```shell
python tools/mc_demo_yolov7.py --mode tracking --source {path_to_video} --project {path_to_save_project}
```

## Acknowledgement

* Thanks to `YOLOv7`, `YOLOv7-pose`, and `BoT-SORT` 
* You can find official github codes in below links

**YOLOv7**

* Official code
* [https://github.com/WongKinYiu/yolov7](https://github.com/WongKinYiu/yolov7)

**YOLOv7-pose**

* Official code
* [https://github.com/WongKinYiu/yolov7/tree/pose](https://github.com/WongKinYiu/yolov7/tree/pose) 

**BoT-SORT**

* Official code
* [https://github.com/NirAharon/BoT-SORT](https://github.com/NirAharon/BoT-SORT) 









