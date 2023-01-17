import argparse

import numpy as np

from typing import Optional
import json

class PoseSimilarity():
    def __init__(self, poses1: dict, poses2: dict):
        self.pose_dict1 = poses1
        self.pose_dict2 = poses2
        self.similarity_scores = None

    def calculate_score(self, distance: Optional[str] = None, score: Optional[str] = None) -> dict:
        assert distance in [None, 'euclidean', 'cosine', 'weighted'], 'Invalid distance method'
        assert score in [None, 'simple', 'sqrt', 'log', 'exp', 'reciprocal'], 'Invalid score method'

        pose_dict1 = self._reformat_poses(self.pose_dict1)
        pose_dict2 = self._reformat_poses(self.pose_dict2)

        pose_scores = {}
        # 각 object에 대해, 
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
                    keypoints = pose_dict2[track_id][frame_id] # There could be non-existing or empty case
                    pose_vector_xy2, pose_vector_score2 = self._flatten_poses(keypoints)
                except:
                    continue
                # Keypoint normalization
                if pose_vector_xy1 == [] or pose_vector_xy2 == []:
                    print('track:', track_id)
                    print('frame:', frame_id)
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
            score = sqrt(1 - distance, 2)
        elif method == 'log':
            score = max(-np.log(distance), 1.0)
        elif method == 'exp':
            score = np.exp(-x), 1
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
    def __init__(self, bboxes1: dict, bboxes2: dict):
        pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Ex command. python tools/scoring.py --pose1 runs/230116/exp23/kpt_result.json --pose2 runs/230116/exp23/kpt_result.json --distance weighted --score simple
    # For calculating pose similarity
    parser.add_argument('--pose1', type=str, default=None, help='Path to pose json file 1')
    parser.add_argument('--pose2', type=str, default=None, help='Path to pose json file 2')
    parser.add_argument('--distance', type=str, default=None, help='[None, euclidean, cosine, weighted]')
    parser.add_argument('--score', type=str, default=None, help='[None, simple, sqrt, log, exp, reciprocal]')
    # For calculating movement similarity
    parser.add_argument('--track1', type=str, default=None, help='Path to track json file 1')
    parser.add_argument('--track2', type=str, default=None, help='Path to track json file 2')

    opt = parser.parse_args()
    print(opt)

    # Run algorithm
    if opt.pose1 and opt.pose2:
        print('Calculate pose similarity...')
        with open(opt.pose1, 'r') as f:
            poses1 = json.load(f)
        with open(opt.pose2, 'r') as f:
            poses2 = json.load(f)
        pose_similarity = PoseSimilarity(poses1, poses2)
        pose_scores = pose_similarity.calculate_score(distance=opt.distance, score=opt.score)
        print('[Pose score]')
        for track_id, pose_score in pose_scores.items():
            print(f'Person {track_id}: {round(pose_score * 100, 2)}')
    if opt.track1 and opt.track2:
        print('Calculate movement similarity...')