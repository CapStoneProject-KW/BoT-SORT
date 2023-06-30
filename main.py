from typing import Optional
import json
import os
import cv2

from fastapi import FastAPI
from pydantic import BaseModel
from server.model import run_model
from server.scoring import run_scoring
from pathlib import Path
import pprint
from yolov7.utils.general import increment_path



class Item(BaseModel):
    user_or_answer: str
    mode: str
    data_path: str

class TrackItem(BaseModel):
    user_or_answer: str
    data_path: str
    ckpt_path: str

class ScoringItem(BaseModel):
    user_kpt_result: str
    answer_kpt_result: str
    user_mot_result: str
    answer_mot_result: str

class ResultItem(BaseModel):
    scoring_path: str
    user_video_path: str
    answer_video_path: str

app = FastAPI()

"""
command: 
uvicorn main:app --reload
"""


@app.get("/")
def main():
    return "Hello World"

@app.get("/items/{item_id}")
def read_item(item_id: int, q: Optional[str] = None):
    return {"item_id": item_id, "q": q}

@app.post("/detect_video")
async def detect_video(item: Item):
    '''
    Descripiton: 
    Args: mode, data_path
    Return: detection result of first frame 
    '''
    # TODO: 이미지 저장 경로 인자로 받기, user인지 answer인지

    # body value from request
    item_dict = item.dict()
    print(item_dict)
    
    # parsing body value
    #mode = item_dict["mode"]
    user_or_answer = item_dict["user_or_answer"]
    data_path = item_dict["data_path"]
    # ckpt_path = item_dict["ckpt_path"]

    # make save directory
    s = list(os.path.split(data_path))
    json_save_path = s[:-1][0]+f'/{user_or_answer}_det_result.json' # data_path + det_result.json
    img_save_path = s[:-1][0]+f'/{user_or_answer}_det_img.jpg'

    # save first frame image for purpose of giving server
    cap = cv2.VideoCapture(data_path)
    if not cap.isOpened(): print("Error opening video file")

    ret, frame = cap.read()
    if not ret: print("Error reading frame")
    
    cv2.imwrite(img_save_path, frame)
    cap.release()
    cv2.destroyAllWindows()
    print('저장 완료')

    # run model
    det_result = run_model('detection', data_path)
    print('[Detection Result]')
    pprint.pprint(det_result)
   
    # save result
    with open(json_save_path, 'w') as f:
        json.dump(det_result[1], f, indent=4)

    print("완료")

    result = {
        "msg": "완료되었습니다.",
        "json_save_path": json_save_path,
        "img_save_path": img_save_path,
        # "det_result": det_result,
    }
    
    # return result
    return result

@app.post("/tracking")
async def tracking_video(item: TrackItem):
    '''
    Descripiton:
    Args: mode, data_path, ckpt_path
    Return:
    '''

    # TODO: 유저에 대한 것과 정답에 대한 파일을 저장할 때 이름을 어떻게 할 지 
    # -> 백엔드에서 넘어올 때 이게 user인지 answer인지 알려줘야 할 듯

    item_dict = item.dict()
    print(item_dict)
    
    user_or_answer = item_dict["user_or_answer"]
    data_path = item_dict["data_path"]
    ckpt_path = item_dict["ckpt_path"]
    
    # make save directory
    if user_or_answer == 'user':
        s = list(os.path.split(data_path))
        kpt_save_path = s[:-1][0]+'/user_kpt_result.json' # data_path + user_kpt_result.json
        mot_save_path = s[:-1][0]+'/user_mot_result.json' # data_path + user_mot_result.json
    elif user_or_answer == 'answer':
        s = list(os.path.split(data_path))
        kpt_save_path = s[:-1][0]+'/answer_kpt_result.json' # data_path + answer_kpt_result.json
        mot_save_path = s[:-1][0]+'/answer_mot_result.json' # data_path + answer_mot_result.json

    # run model
    kpt_result, mot_result = run_model('tracking', data_path)

    # print('[Keypoint Result]')
    # pprint.pprint(print_kpt)
    # print('[Tracking Result]')
    # pprint.pprint(print_mot)

    with open(kpt_save_path, 'w') as f:
        json.dump(kpt_result, f, indent=4)

    with open(mot_save_path, 'w') as f:
        json.dump(mot_result, f, indent=4)

    print("완료")

    # make result
    # result = [
    #     kpt_result, 
    #     mot_result
    # ]
    result = {
        "msg": "완료되었습니다.",
        "kpt_save_path": kpt_save_path,
        "mot_save_path": mot_save_path
    }
    
    return result

@app.post("/scoring")
async def scoring(item: ScoringItem):
    '''
    Descripiton:
    Args: user_kpt_result, answer_kpt_result, user_mot_result, answer_mot_result
    Return: Pose Score, Movement Score each track_id
    '''
    # TODO: frame 별 score -> 저장 경로 지정
    
    # run_model("detection", "../datasets/last_dance/user_video.mp4", "../pretrained/yolov7.pt")
    item_dict = item.dict()
    print(item_dict)
    
    # parsing body value
    user_kpt_result = item_dict["user_kpt_result"]
    answer_kpt_result = item_dict["answer_kpt_result"]
    user_mot_result = item_dict["user_mot_result"]
    answer_mot_result = item_dict["answer_mot_result"]

    # loading 4 json files
    with open(user_kpt_result, 'r') as f:
        user_kpt_result_dict = json.load(f)
    with open(answer_kpt_result, 'r') as f:
        answer_kpt_result_dict = json.load(f)
    with open(user_mot_result, 'r') as f:
        user_mot_result_dict = json.load(f)
    with open(answer_mot_result, 'r') as f:
        answer_mot_result_dict = json.load(f)


    matches = [['1', '1']]
    # run scoring function
    pose_scores, score_per_pose = run_scoring('pose',                       
                            user_kpt_result_dict,
                            answer_kpt_result_dict, 
                            matches=matches,
                            distance='euclidean', 
                            score='simple')
    movement_scores, score_per_move = run_scoring('movement', 
                            user_mot_result_dict, 
                            answer_mot_result_dict, 
                            matches=matches,
                            distance='euclidean', 
                            score='simple')

    print('[Pose Score]')
    for track_id, score in pose_scores.items():
        print(f"ID: {track_id} Score: {score}")

    print('[Movement Score]')
    for track_id, score in movement_scores.items():
        print(f"ID: {track_id} Score: {score}")

    # make return result
    result = {
        "msg": "완료되었습니다.",
        "pose_scores": pose_scores,
        "movement_scores": movement_scores
    }
    
    return result

@app.post("/result_video")
async def scoring(item: ScoringItem):
    '''
    Descripiton:
    Args: user_kpt_result, answer_kpt_result, user_mot_result, answer_mot_result
    Return: Pose Score, Movement Score each track_id
    '''
    
