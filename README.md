# MOT-YOLOv7-pose (full ver.)

## Introduction

_TBU_ 

## NOTE) How to use?

**Environment setup**

* Use `my_requirements.yaml` file
* NOTE) You need to install appropriate version of cuda(11.6) and cudnn first
* NOTE) If any problem occurs, please check [here](https://github.com/NirAharon/BoT-SORT#installation) for further information

```shell
conda env create -f my_requirements.yaml
```

Then, run `setup.py` to setup environment.

```shell
python3 setup.py develop
```

For last, create `pretrained` folder and download [yolov7.pt](https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7.pt) and [yolov7-w6-pose.pt](https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7-w6-pose.pt) weight files in it. 


**Run model**

You can run **detection mode** and **tracking mode** using our model. 

* Detection mode (Only detection for the first frame of the video)

```shell
python tools/mc_demo_yolov7.py --mode detection --source {path_to_video} --project {path_to_save_project}
```

* Tracking mode (Keypoint extraction + Multiple object tracking for the entire length of the video)

```shell
python tools/mc_demo_yolov7.py --mode tracking --source {path_to_video} --project {path_to_save_project}
```

After you run the model, you can find the results at `{path_to_save_project}` directory 

## Result format

**Detection result**

```json
{
    "1": { # bbox id
        "x1": 89.73,
        "y1": 272.25,
        "w": 127.05,
        "h": 345.38,
        "s": 0.91
    },
    "2": {
        "x1": 194.0,
        "y1": 271.12,
        "w": 100.69,
        "h": 355.5,
        "s": 0.92
    }
}
```



**Keypoint extraction result**

```json
{
    "0": { # frame id
        "1": { # track id
            "0": { # keypoint id
                "x": 240.27,
                "y": 290.25,
                "s": 0.99
            },
            "1": {
                "x": 247.44,
                "y": 284.34,
                "s": 0.99
            },
            ..., 
            "16": { # 17 keypoints
                "x": 231.97,
                "y": 599.06,
                "s": 0.95
            }
        },
        ..., 
    }, 
    ..., 
}
```



**MOT result**

```json
{
    "0": { # frame id
        "1": { # track id
            "x1": 196.78,
            "y1": 264.36,
            "w": 94.85,
            "h": 368.44,
            "s": 0.89
        },
        ..., 
    },
    ..., 
}
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









