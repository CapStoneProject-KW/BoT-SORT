import argparse

import numpy as np
from math import sqrt

from typing import Optional
import json

class PoseSimilarity():
    def __init__(self, poses1: dict, poses2: dict, matches: list):
        self.pose_dict1 = poses1 # user
        self.pose_dict2 = poses2 # answer
        self.matches = {}
        for user_id, answer_id in matches:
            self.matches[user_id] = answer_id
        self.similarity_scores = None

    def calculate_score(self, distance: Optional[str] = None, score: Optional[str] = None) -> dict:
        # distance와 score가 리스트안의 원소 값을 가져야 함을 보장
        assert distance in [None, 'euclidean', 'cosine', 'weighted'], 'Invalid distance method'
        assert score in [None, 'simple', 'sqrt', 'log', 'exp', 'reciprocal'], 'Invalid score method'

        # print(self.pose_dict1)
        # print(self.pose_dict2)

        # 각 tracking object에 대해서 프레임마다 바뀌는 keypoint들 좌표 - 제일 상위 key가 track_id(... -> frame_id -> keypoints)
        pose_dict1 = self._reformat_poses(self.pose_dict1)
        pose_dict2 = self._reformat_poses(self.pose_dict2)

        # print(json.dumps(pose_dict1, ensure_ascii=False, indent=3))
        # print(json.dumps(pose_dict2, ensure_ascii=False, indent=3))

        pose_scores = {}
        # 각 object에 대해, [key, value]
        for track_id, frames in pose_dict1.items():
            # 각 frame에 대해, 
            pose_score_sum = 0
            valid_frame = 0
            for frame_id, keypoints in frames.items():
                # keypoints 정보를 flatten
                pose_vector_xy1, pose_vector_score1 = self._flatten_poses(keypoints)
                # 이걸 pose 2개에 대해 하고, 
                # 각 frame의 similarity를 구해서
                try:
                    match_id = self.matches[track_id]
                    keypoints = pose_dict2[match_id][frame_id] # There could be non-existing or empty case
                    pose_vector_xy2, pose_vector_score2 = self._flatten_poses(keypoints)
                except:
                    continue
                # Keypoint normalization
                if pose_vector_xy1 == [] or pose_vector_xy2 == [] or len(pose_vector_xy1) != len(pose_vector_xy2):
                    # print('track:', track_id)
                    # print('match:', match_id)
                    # print('frame:', frame_id)
                    continue
                pose_vector_xy1 = self._normalize_poses(pose_vector_xy1)
                pose_vector_xy2 = self._normalize_poses(pose_vector_xy2)
                # Calculate the distance between two poses (0~1)
                pose_distance = self._get_distance(pose_vector_xy1, pose_vector_xy2, weight=pose_vector_score1, method=distance)
                # Convert distance to similarity score (0~1)
                pose_score = self._get_score(pose_distance, method=score)
                pose_score_sum += pose_score
                valid_frame += 1
            # 모든 frame의 similarity 정보들을 이용해 해당 object의 score를 매긴다.
            if valid_frame == 0:
                continue
            # print(pose_score_sum, valid_frame) 
            pose_scores[track_id] = pose_score_sum / valid_frame
        # object scores 반환
        self.similarity_scores = pose_scores
        return pose_scores

    def _get_distance(self, vec1: list, vec2: list, weight=None, method=None) -> float:
        distance = 0
        if method in ['euclidean', None]:
            distance = sqrt(sum([(x - y) ** 2 for x, y in zip(vec1, vec2)]))
        elif method == 'cosine':
            cossim = np.dot(vec1, vec2) / (np.linalg.norm(vec1, 2) * np.linalg.norm(vec2, 2))
            distance = sqrt(2 * (1 - cossim))
        elif method == 'weighted':
            distance = sum([weight[i//2] * ((vec1[i] - vec2[i])**2) for i in range(len(vec1))]) / weight[-1]

        assert 0 <= distance <= 1, 'Distance out of boundary'
        return distance

    def _get_score(self, distance: float, method=None) -> float:
        if method in ['simple', None]:
            score = 1 - distance
        elif method == 'sqrt':
            score = sqrt(1 - distance)
        elif method == 'log':
            score = max(-np.log(distance), 1.0)
        elif method == 'exp':
            score = np.exp(-distance), 1
        elif method == 'reciprocal':
            score = 1 / (1 + distance)

        assert 0 <= score <= 1, 'Score out of boundary'
        return score

    def _normalize_poses(self, pose_vector_xy: list) -> list:
        # Subtraction
        x_min, y_min = min(pose_vector_xy[::2]), min(pose_vector_xy[1::2])
        normalized_xy = [pose_vector_xy[i] - x_min if i % 2 == 0 else pose_vector_xy[i] - y_min for i in range(len(pose_vector_xy))]
        # Scaling
        xy_max = max(normalized_xy)
        normalized_xy = [xy / xy_max for xy in normalized_xy]
        # L2 normalize
        xy_l2norm = np.linalg.norm(normalized_xy, 2)
        normalized_xy = [xy / xy_l2norm for xy in normalized_xy]

        return normalized_xy

    def _reformat_poses(self, poses:dict) -> dict:
        """
        Description:
            frame_id / track_id / kpt_id -> track_id / frame_id / kpt_id
        Ret:
            Reformatted pose dictionary
        """
        reformatted_poses = {}
        # For each frame, 
        for frame_id, tracks in poses.items():
            # for each object, 
            for track_id, keypoints in tracks.items():
                if track_id not in reformatted_poses:
                    reformatted_poses[track_id] = {}
                reformatted_poses[track_id][frame_id] = keypoints
        return reformatted_poses
    
    def _flatten_poses(self, keypoints: dict) -> [list, list]:
        """
        Description:
            Convert keypoints(dict) to pose vectors(list)
        Ret:
            pose_vector_xy: x and y coordinate for each keypoint (len: 2n)
            pose_vector_score: score for each keypoint and sum of the scores (len: n+1)
        """
        pose_vector_xy = []
        pose_vector_score = []
        for keypoint_id, keypoint in keypoints.items():
            pose_vector_xy.append(keypoint["x"])
            pose_vector_xy.append(keypoint["y"])
            pose_vector_score.append(keypoint["s"])
        pose_vector_score.append(sum(pose_vector_score))

        return pose_vector_xy, pose_vector_score


class MovementSimilarity():
    def __init__(self, bboxes1: dict, bboxes2: dict, matches: list):
        self.bboxes_dict1 = bboxes1 # user
        self.bboxes_dict2 = bboxes2 # answer
        self.matches = {}
        for user_id, answer_id in matches:
            self.matches[user_id] = answer_id
        self.similarity_scores = None

    def calculate_score(self, distance: Optional[str] = None, score: Optional[str] = None) -> dict:
        # distance와 score가 리스트안의 원소 값을 가져야 함을 보장
        assert distance in [None, 'euclidean', 'cosine', 'weighted'], 'Invalid distance method'
        assert score in [None, 'simple', 'sqrt', 'log', 'exp', 'reciprocal'], 'Invalid score method'

        # print(self.bboxes_dict1)
        # print(self.bboxes_dict2)

        # 각 tracking object에 대해서 프레임마다 바뀌는 keypoint들 좌표 - 제일 상위 key가 track_id(... -> frame_id -> keypoints)
        bboxes_dict1 = self._reformat_bboxes(self.bboxes_dict1)
        bboxes_dict2 = self._reformat_bboxes(self.bboxes_dict2)

        # print(json.dumps(bboxes_dict1, ensure_ascii=False, indent=3))
        # print(json.dumps(bboxes_dict2, ensure_ascii=False, indent=3))
        
        movement_scores = {}
        # 각 object에 대해, [key, value]
        for track_id, frames in bboxes_dict1.items():
            # 각 frame에 대해, 
            movement_score_sum = 0
            valid_frame = 0

            if track_id not in bboxes_dict2:
                continue

            frame_keys_dict1 = sorted(bboxes_dict1[track_id].keys())
            match_id = self.matches[track_id]
            frame_keys_dict2 = sorted(bboxes_dict2[match_id].keys())
            # print(frame_keys_dict1, frame_keys_dict2)

            # for i in range(len(frame_keys_dict1)):
            #     x1, y1 = normalized_video_A[keys_video_A[i]]
            #     x2, y2 = normalized_video_A[keys_video_A[i + 1]]
            #     differences_video_A.append(calculate_difference(x1, y1, x2, y2))

            for i, (frame_id, bbox) in enumerate(frames.items()):
                '''
                {
                    "x1": 152.7,
                    "y1": 203.24,
                    "w": 110.26,
                    "h": 355.43,
                    "s": 0.91
                }
                '''
                # coordinate 정보를 flatten
                '''
                [157.56, 193.32]
                [92.81, 346.5]
                [0.93, 0.93]
                '''
                
                if i == len(frame_keys_dict1) - 1:
                    break

                # bbox_vector_xy1, bbox_vector_wh1, bbox_vector_score1 = self._flatten_bboxes(bbox)
                bbox_vector_xy1, bbox_vector_wh1, bbox_vector_score1 = self._flatten_bboxes(frames[frame_keys_dict1[i]])
                next_bbox_vector_xy1, next_bbox_vector_wh1, next_bbox_vector_score1 = self._flatten_bboxes(frames[frame_keys_dict1[i+1]])


                # print(bbox_vector_xy1, bbox_vector_wh, bbox_vector_score)
                
                # 이걸 pose 2개에 대해 하고, 
                # 각 frame의 similarity를 구해서
                try:
                    bbox = bboxes_dict2[match_id][frame_id] # There could be non-existing or empty case
                    
                    bbox_vector_xy2, bbox_vector_wh2, bbox_vector_score2 = self._flatten_bboxes(frames[frame_keys_dict2[i]])
                    next_bbox_vector_xy2, next_bbox_vector_wh2, next_bbox_vector_score2 = self._flatten_bboxes(frames[frame_keys_dict2[i+1]])
                except:
                    continue

                # bbox normalization
                if bbox_vector_xy1 == [] or bbox_vector_xy2 == [] or len(bbox_vector_xy1) != len(bbox_vector_xy1) \
                    or next_bbox_vector_xy1 == [] or next_bbox_vector_xy2 == [] or len(next_bbox_vector_xy1) != len(next_bbox_vector_xy1):
                    # print('track:', track_id)
                    # print('frame:', frame_id)
                    continue

                bbox_vector_xy1 = self._scaling_coordinate(bbox_vector_xy1, bbox_vector_wh1)
                next_bbox_vector_xy1 = self._scaling_coordinate(next_bbox_vector_xy1, next_bbox_vector_wh1)

                bbox_vector_xy2 = self._scaling_coordinate(bbox_vector_xy2, bbox_vector_wh2)
                next_bbox_vector_xy2 = self._scaling_coordinate(next_bbox_vector_xy2, next_bbox_vector_wh2)

                # print(bbox_vector_xy1, next_bbox_vector_xy1) # [1.6976618898825557, 0.5579220779220779] [1.3762291169451073, 0.5418694551825792]
                # print(bbox_vector_xy2, next_bbox_vector_xy2) # [1.6976618898825557, 0.5579220779220779] [1.3762291169451073, 0.5418694551825792]

                # Calcuate the difference between (n)th frame and (n+1)th frame
                differences_bbox_vector_xy1 = [next_bbox_vector_xy1[0] - bbox_vector_xy1[0], next_bbox_vector_xy1[1] - bbox_vector_xy1[1]]
                differences_bbox_vector_xy2 = [next_bbox_vector_xy2[0] - bbox_vector_xy2[0], next_bbox_vector_xy2[1] - bbox_vector_xy2[1]]
                # print(differences_bbox_vector_xy1, differences_bbox_vector_xy2)
                
                # Calculate the distance between two poses (0~1)
                bbox_distance = self._get_distance(differences_bbox_vector_xy1, differences_bbox_vector_xy2, weight=bbox_vector_score1, method=distance)
                # print("b_d", bbox_distance)

                # Convert distance to similarity score (0~1)
                bbox_score = self._get_score(bbox_distance, method=score)
                # print(bbox_score)
                movement_score_sum += bbox_score
                valid_frame += 1
            # 모든 frame의 similarity 정보들을 이용해 해당 object의 score를 매긴다.
            # print(movement_score_sum, valid_frame) 
            if valid_frame == 0:
                continue
            movement_scores[track_id] = movement_score_sum / valid_frame
        # object scores 반환
        self.similarity_scores = movement_scores
        return movement_scores

    def _get_distance(self, vec1: list, vec2: list, weight=None, method=None) -> float:
        distance = 0
        if method in ['euclidean', None]:
            distance = sqrt(sum([(x - y) ** 2 for x, y in zip(vec1, vec2)]))
        elif method == 'cosine':
            cossim = np.dot(vec1, vec2) / (np.linalg.norm(vec1, 2) * np.linalg.norm(vec2, 2))
            distance = sqrt(2 * (1 - cossim))
        elif method == 'weighted':
            distance = sum([weight[i//2] * ((vec1[i] - vec2[i])**2) for i in range(len(vec1))]) / weight[-1]

        assert 0 <= distance <= 1, 'Distance out of boundary'
        return distance

    def _get_score(self, distance: float, method=None) -> float:
        if method in ['simple', None]:
            score = 1 - distance
        elif method == 'sqrt':
            score = sqrt(1 - distance)
        elif method == 'log':
            score = max(-np.log(distance), 1.0)
        elif method == 'exp':
            score = np.exp(-distance), 1
        elif method == 'reciprocal':
            score = 1 / (1 + distance)

        assert 0 <= score <= 1, 'Score out of boundary'
        return score
    

    def _normalize_poses(self, pose_vector_xy: list) -> list:
        # Subtraction
        x_min, y_min = min(pose_vector_xy[::2]), min(pose_vector_xy[1::2])
        normalized_xy = [pose_vector_xy[i] - x_min if i % 2 == 0 else pose_vector_xy[i] - y_min for i in range(len(pose_vector_xy))]
        # Scaling
        xy_max = max(normalized_xy)
        normalized_xy = [xy / xy_max for xy in normalized_xy]
        # L2 normalize
        xy_l2norm = np.linalg.norm(normalized_xy, 2)
        normalized_xy = [xy / xy_l2norm for xy in normalized_xy]

        return normalized_xy

    def _scaling_coordinate(self, bbox_vector_xy: list, bbox_vector_wh: list) -> list:
        """
        Description:
            Scaling coordinate(dict) to proportional dict(list)
        """
        '''
            [157.56, 193.32]
            [92.81, 346.5]
        '''
        # Subtraction
        x, y = bbox_vector_xy[0], bbox_vector_xy[1]
        w, h = bbox_vector_wh[0], bbox_vector_wh[1]

        normalized_xy = [x/w, y/h]

        return normalized_xy


    def _reformat_bboxes(self, bboxes:dict) -> dict:
        """
        Description:
            frame_id / track_id / kpt_id -> track_id / frame_id / kpt_id
        Ret:
            Reformatted pose dictionary
        """
        reformatted_bboxes = {}
        # For each frame, 
        for frame_id, tracks in bboxes.items():
            # for each object, 
            for track_id, coordinate in tracks.items():
                if track_id not in reformatted_bboxes:
                    reformatted_bboxes[track_id] = {}
                reformatted_bboxes[track_id][frame_id] = coordinate
        return reformatted_bboxes

    def _flatten_bboxes(self, bboxes: dict) -> [list, list]:
        """
        Description:
            Convert keypoints(dict) to pose vectors(list)
        Ret:
            bbox_vector_xy: x and y coordinate for each keypoint (len: 2n)
            bbox_vector_score: score for each keypoint and sum of the scores (len: n+1)
        """
        bbox_vector_xy = []
        bbox_vector_wh = []
        bbox_vector_score = []
        for bbox_id, bbox in bboxes.items():
            if bbox_id == "x1" or bbox_id == "y1":
                bbox_vector_xy.append(bbox)
            elif bbox_id == "w" or bbox_id == "h":
                bbox_vector_wh.append(bbox)
            elif bbox_id == "s":
                bbox_vector_score.append(bbox)

            
        bbox_vector_score.append(sum(bbox_vector_score))

        return bbox_vector_xy, bbox_vector_wh, bbox_vector_score


def run_scoring(mode, dict1, dict2, matches, distance=None, score=None):

    matches = [[1, 1], [2, 2], [3, 3], [4, 4], [5, 5]]
    if mode == 'pose':
        similarity_obj = PoseSimilarity(dict1, dict2, matches)
    elif mode == 'movement':
        similarity_obj = MovementSimilarity(dict1, dict2, matches)

    scores = similarity_obj.calculate_score(distance=distance, score=score)
    return scores