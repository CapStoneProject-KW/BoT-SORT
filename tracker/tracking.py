import cv2
import matplotlib.pyplot as plt
import numpy as np
from collections import deque

from tracker import matching
from tracker.gmc import GMC
from tracker.basetrack import BaseTrack, TrackState
from tracker.kalman_filter import KalmanFilter

from fast_reid.fast_reid_interfece import FastReIDInterface


class STrack(BaseTrack):
    shared_kalman = KalmanFilter()

    def __init__(self, frame_id, tlwh, score, cls, kpt, feat=None, feat_history=50):

        # wait activate
        self._tlwh = np.asarray(tlwh, dtype=np.float)
        self.kalman_filter = None
        self.mean, self.covariance = None, None
        self.is_activated = False

        self.cls = -1
        self.cls_hist = []  # (cls id, freq)
        self.update_cls(cls, score)

        self.score = score
        self.tracklet_len = 0

        self.smooth_feat = None
        self.curr_feat = None
        if feat is not None:
            self.update_features(feat)
        self.features = deque([], maxlen=feat_history)
        self.alpha = 0.9

        self.kpts = {frame_id: kpt} # int, list (shape: (17, 3))

    def update_kpts(self, kpts: dict):
        # TODO: update frame keypoints
        for frame_id, kpt in list(kpts.items()):
            if frame_id not in self.kpts:
                self.kpts[frame_id] = kpt

    def update_features(self, feat):
        feat /= np.linalg.norm(feat)
        self.curr_feat = feat
        if self.smooth_feat is None:
            self.smooth_feat = feat
        else:
            self.smooth_feat = self.alpha * self.smooth_feat + (1 - self.alpha) * feat
        self.features.append(feat)
        self.smooth_feat /= np.linalg.norm(self.smooth_feat)

    def update_cls(self, cls, score):
        if len(self.cls_hist) > 0:
            max_freq = 0
            found = False
            for c in self.cls_hist:
                if cls == c[0]:
                    c[1] += score
                    found = True

                if c[1] > max_freq:
                    max_freq = c[1]
                    self.cls = c[0]
            if not found:
                self.cls_hist.append([cls, score])
                self.cls = cls
        else:
            self.cls_hist.append([cls, score])
            self.cls = cls

    def predict(self):
        mean_state = self.mean.copy()
        if self.state != TrackState.Tracked:
            mean_state[6] = 0
            mean_state[7] = 0

        self.mean, self.covariance = self.kalman_filter.predict(mean_state, self.covariance)

    @staticmethod
    def multi_predict(stracks):
        if len(stracks) > 0:
            multi_mean = np.asarray([st.mean.copy() for st in stracks])
            multi_covariance = np.asarray([st.covariance for st in stracks])
            for i, st in enumerate(stracks):
                if st.state != TrackState.Tracked:
                    multi_mean[i][6] = 0
                    multi_mean[i][7] = 0
            multi_mean, multi_covariance = STrack.shared_kalman.multi_predict(multi_mean, multi_covariance)
            for i, (mean, cov) in enumerate(zip(multi_mean, multi_covariance)):
                stracks[i].mean = mean
                stracks[i].covariance = cov

    @staticmethod
    def multi_gmc(stracks, H=np.eye(2, 3)):
        if len(stracks) > 0:
            multi_mean = np.asarray([st.mean.copy() for st in stracks])
            multi_covariance = np.asarray([st.covariance for st in stracks])

            R = H[:2, :2]
            R8x8 = np.kron(np.eye(4, dtype=float), R)
            t = H[:2, 2]

            for i, (mean, cov) in enumerate(zip(multi_mean, multi_covariance)):
                mean = R8x8.dot(mean)
                mean[:2] += t
                cov = R8x8.dot(cov).dot(R8x8.transpose())

                stracks[i].mean = mean
                stracks[i].covariance = cov

    def activate(self, kalman_filter, frame_id):
        """Start a new tracklet"""
        self.kalman_filter = kalman_filter
        self.track_id = self.next_id()

        self.mean, self.covariance = self.kalman_filter.initiate(self.tlwh_to_xywh(self._tlwh))

        self.tracklet_len = 0
        self.state = TrackState.Tracked
        if frame_id == 1:
            self.is_activated = True
        self.frame_id = frame_id
        self.start_frame = frame_id

    def re_activate(self, new_track, frame_id, new_id=False):

        self.mean, self.covariance = self.kalman_filter.update(self.mean, self.covariance, self.tlwh_to_xywh(new_track.tlwh))
        if new_track.curr_feat is not None:
            self.update_features(new_track.curr_feat)
        self.tracklet_len = 0
        self.state = TrackState.Tracked
        self.is_activated = True
        self.frame_id = frame_id
        if new_id:
            self.track_id = self.next_id()
        self.score = new_track.score

        self.update_cls(new_track.cls, new_track.score)

    def update(self, new_track, frame_id):
        """
        Update a matched track
        :type new_track: STrack
        :type frame_id: int
        :type update_feature: bool
        :return:
        """
        self.frame_id = frame_id
        self.tracklet_len += 1

        new_tlwh = new_track.tlwh

        self.mean, self.covariance = self.kalman_filter.update(self.mean, self.covariance, self.tlwh_to_xywh(new_tlwh))

        if new_track.curr_feat is not None:
            self.update_features(new_track.curr_feat)

        self.state = TrackState.Tracked
        self.is_activated = True

        self.score = new_track.score
        self.update_cls(new_track.cls, new_track.score)

        # update keypoints
        self.update_kpts(new_track.kpts)

    @property
    def tlwh(self):
        """Get current position in bounding box format `(top left x, top left y,
                width, height)`.
        """
        if self.mean is None:
            return self._tlwh.copy()
        ret = self.mean[:4].copy()
        ret[:2] -= ret[2:] / 2
        return ret

    @property
    def tlbr(self):
        """Convert bounding box to format `(min x, min y, max x, max y)`, i.e.,
        `(top left, bottom right)`.
        """
        ret = self.tlwh.copy()
        ret[2:] += ret[:2]
        return ret

    @property
    def xywh(self):
        """Convert bounding box to format `(min x, min y, max x, max y)`, i.e.,
        `(top left, bottom right)`.
        """
        ret = self.tlwh.copy()
        ret[:2] += ret[2:] / 2.0
        return ret

    @property
    def frame_kpts(self):
        """Save keypoint coordinates for each frame
        """


    @staticmethod
    def tlwh_to_xyah(tlwh):
        """Convert bounding box to format `(center x, center y, aspect ratio,
        height)`, where the aspect ratio is `width / height`.
        """
        ret = np.asarray(tlwh).copy()
        ret[:2] += ret[2:] / 2
        ret[2] /= ret[3]
        return ret

    @staticmethod
    def tlwh_to_xywh(tlwh):
        """Convert bounding box to format `(center x, center y, width,
        height)`.
        """
        ret = np.asarray(tlwh).copy()
        ret[:2] += ret[2:] / 2
        return ret

    def to_xywh(self):
        return self.tlwh_to_xywh(self.tlwh)

    @staticmethod
    def tlbr_to_tlwh(tlbr):
        ret = np.asarray(tlbr).copy()
        ret[2:] -= ret[:2]
        return ret

    @staticmethod
    def tlwh_to_tlbr(tlwh):
        ret = np.asarray(tlwh).copy()
        ret[2:] += ret[:2]
        return ret

    def __repr__(self):
        return 'OT_{}_({}-{})'.format(self.track_id, self.start_frame, self.end_frame)


class BoTSORT(object):
    def __init__(self, args, frame_rate=30):
        
        ### Tracker
        # existing track pool
        self.strack_pool = []  # type: list[STrack]
        # losted tracks
        self.lost_stracks = []

        BaseTrack.clear_count()

        self.frame_id = -1
        self.total_obj = None
        self.args = args

        self.track_high_thresh = args.track_high_thresh
        self.track_low_thresh = args.track_low_thresh
        self.new_track_thresh = args.new_track_thresh

        # Buffer size (second) when FPS = 30
        self.buffer_size = int(frame_rate / 30.0 * args.track_buffer)
        # Considered lost if the object is not tracked for a certain period of time
        self.max_time_lost = self.buffer_size
        # Kalman filter object
        self.kalman_filter = KalmanFilter()

        ### ReID module
        # theta_iou: Set to 0.5 and eliminate unlikely pairs between tracklets and detections (12)
        self.proximity_thresh = args.proximity_thresh
        # theta_emb: Set to 0.25 and separate positive association of tracklet appearance states and detections embedding vectors from the negative ones
        self.appearance_thresh = args.appearance_thresh
        # flag: use Re-ID module
        if args.with_reid:
            print(args.device)
            self.encoder = FastReIDInterface(args.fast_reid_config, args.fast_reid_weights, args.device)
        # Global motion compensation (GMC) technique for camera motion compensation (CMC)
        self.gmc = GMC(method=args.cmc_method, verbose=[args.name, args.ablation])

    def update(self, output_results, img):
        '''
        Description:
        Args:
            output_results: scaled detection bboxes + keypoints
            img: origin frame image
        Ret:
            output_stracks: online trackers for current frame
        '''
        # frame id
        self.frame_id += 1
        # status: tracked -> tracked
        activated_stracks = []
        # status: untracked -> tracked
        refind_stracks = []

        # Fix total object count to object count in initial detection
        if self.frame_id == 0:
            self.total_obj = len(output_results)

        # detection exists
        if len(output_results):
            # x, y, w, h
            bboxes = output_results[:, :4]
            dets = output_results[:, :4]
            # confidence score
            scores = output_results[:, 4]
            # class
            classes = output_results[:, 5]
            # 17 keypoints
            kpts = output_results[:, 6:]
            # ReID features
            features = output_results[:, 6:]
        # no detection
        else:
            bboxes = []
            dets = []
            scores = []
            classes = []
            kpts = []
            features = []

        ''' 1. Extract embeddings of detected objects '''
        # extracted features for detected bboxes with ReID module
        if self.args.with_reid:
            features = self.encoder.inference(img, dets)
        # STracks of detected bboxes
        detected_stracks = []
        for tlbr, s, c, k, f in zip(dets, scores, classes, kpts, features):
            if self.args.with_reid:
                track = STrack(self.frame_id, STrack.tlbr_to_tlwh(tlbr), s, c, k, f)
            else:
                track = STrack(self.frame_id, STrack.tlbr_to_tlwh(tlbr), s, c, k)
            if self.frame_id == 0:
                track.activate(self.kalman_filter, self.frame_id)
                self.strack_pool.append(track)
            detected_stracks.append(track)

        # print()
        # print(f'Frame {self.frame_id}')
        
        # print('detected_stracks')
        # print(detected_stracks)

        ''' 2. Get activated/non-activated tracks from track pool '''
        unconfirmed_stracks = []
        # stracks for current frame
        tracked_stracks = [] # type: list[STrack]
        # existing track pool
        for track in self.strack_pool:
            # deactivated tracks
            if not track.is_activated:
                unconfirmed_stracks.append(track)
            # activated tracks
            else:
                tracked_stracks.append(track)

        ''' 3. Match track pool with tracks of current frame '''
        # Predict the current location with KF
        # print('self.strack_pool')
        # print(self.strack_pool)

        STrack.multi_predict(self.strack_pool)
        # Fix camera motion
        warp = self.gmc.apply(img, dets)
        STrack.multi_gmc(self.strack_pool, warp)
        STrack.multi_gmc(unconfirmed_stracks, warp)
        # Associate track pool with detection boxes
        ious_dists = matching.iou_distance(self.strack_pool, detected_stracks)
        ious_dists_mask = (ious_dists > self.proximity_thresh)

        # print('ious_dists')
        # print(ious_dists)
        # print('ious_dists_mask')
        # print(ious_dists_mask)

        if not self.args.mot20:
            ious_dists = matching.fuse_score(ious_dists, detected_stracks)
        if self.args.with_reid:
            emb_dists = matching.embedding_distance(self.strack_pool, detected_stracks) / 2.0
            # print('emb_dists')
            # print(emb_dists)
            raw_emb_dists = emb_dists.copy()
            emb_dists[emb_dists > self.appearance_thresh] = 1.0
            emb_dists[ious_dists_mask] = 1.0
            dists = np.minimum(ious_dists, emb_dists)
            # print('emb_dists')
            # print(emb_dists)
        else:
            dists = ious_dists
        # a-b cost matrix -> matches, unmatched_a, unmatched_b 
        # print('dists')
        # print(dists)
        matches, u_track, u_detection = matching.linear_assignment(dists, thresh=self.args.match_thresh) 
        
        # print('matches, u_track, u_detection')
        # print(matches, u_track, u_detection)
        # if self.frame_id > 100:
        #     exit(0)
        ''' 4. Update values of track pool '''
        for itracked, idet in matches:
            # matched tracks
            track = self.strack_pool[itracked]
            # matched detections
            det = detected_stracks[idet]
            # update matched tracks using detections of current frame
            if track.state == TrackState.Tracked:
                track.update(detected_stracks[idet], self.frame_id)
                # the track was activated in advance
                activated_stracks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                # the track wasn't activated
                refind_stracks.append(track)

        ''' 5. Merge strack pool and return '''
        # self.tracked_stracks = [t for t in self.tracked_stracks if t.state == TrackState.Tracked]
        # self.tracked_stracks = joint_stracks(self.frame_id, self.tracked_stracks, activated_stracks, 'call from 2')
        # self.tracked_stracks = joint_stracks(self.frame_id, self.tracked_stracks, refind_stracks, 'call from 3')

        output_stracks = [track for track in self.strack_pool]

        # print('output_stracks')
        # print(output_stracks)
        
        return output_stracks


        """ Deprecated

        ''' Step 1: Add newly detected tracklets to tracked_stracks'''
        unconfirmed = []
        # tracked_stracks: activated tracks in previous frame
        tracked_stracks = []  # type: list[STrack]
        # self.tracked_stracks: existing stracks
        for track in self.tracked_stracks:
            # deactivated tracks
            if not track.is_activated:
                unconfirmed.append(track)
            # activated tracks
            else:
                tracked_stracks.append(track)

        ''' Step 2: First association, with high score detection boxes'''
        # Merge tracked_stracks and self.lost_stracks
        strack_pool = joint_stracks(self.frame_id, tracked_stracks, self.lost_stracks, 'call from 1')

        # Predict the current location with KF
        STrack.multi_predict(strack_pool)

        # Fix camera motion
        warp = self.gmc.apply(img, dets)
        STrack.multi_gmc(strack_pool, warp)
        STrack.multi_gmc(unconfirmed, warp)

        # Associate with high score detection boxes
        ious_dists = matching.iou_distance(strack_pool, detections)
        ious_dists_mask = (ious_dists > self.proximity_thresh)

        if not self.args.mot20:
            ious_dists = matching.fuse_score(ious_dists, detections)

        if self.args.with_reid:
            emb_dists = matching.embedding_distance(strack_pool, detections) / 2.0
            raw_emb_dists = emb_dists.copy()
            emb_dists[emb_dists > self.appearance_thresh] = 1.0
            emb_dists[ious_dists_mask] = 1.0
            dists = np.minimum(ious_dists, emb_dists)

            # Popular ReID method (JDE / FairMOT)
            # raw_emb_dists = matching.embedding_distance(strack_pool, detections)
            # dists = matching.fuse_motion(self.kalman_filter, raw_emb_dists, strack_pool, detections)
            # emb_dists = dists

            # IoU making ReID
            # dists = matching.embedding_distance(strack_pool, detections)
            # dists[ious_dists_mask] = 1.0
        else:
            dists = ious_dists

        matches, u_track, u_detection = matching.linear_assignment(dists, thresh=self.args.match_thresh)

        for itracked, idet in matches:
            # matched tracks
            track = strack_pool[itracked]
            # matched detections
            det = detections[idet]
            # update matched tracks using detections of current frame
            if track.state == TrackState.Tracked:
                track.update(detections[idet], self.frame_id)
                # the track was activated in advance
                activated_stracks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                # the track wasn't activated
                refind_stracks.append(track)

        ''' Step 3: Second association, with low score detection boxes'''
        if len(scores):
            inds_high = scores < self.args.track_high_thresh
            inds_low = scores > self.args.track_low_thresh
            inds_second = np.logical_and(inds_low, inds_high)
            dets_second = bboxes[inds_second]
            scores_second = scores[inds_second]
            classes_second = classes[inds_second]
            kpts_second = kpts[inds_second]
        else:
            dets_second = []
            scores_second = []
            classes_second = []
            kpts_second = []

        # association the untrack to the low score detections
        if len(dets_second) > 0:
            '''Detections'''
            detections_second = [STrack(self.frame_id, STrack.tlbr_to_tlwh(tlbr), s, c, k) for
                                 (tlbr, s, c, k) in zip(dets_second, scores_second, classes_second, kpts_second)]
        else:
            detections_second = []

        r_tracked_stracks = [strack_pool[i] for i in u_track if strack_pool[i].state == TrackState.Tracked]
        dists = matching.iou_distance(r_tracked_stracks, detections_second)
        matches, u_track, u_detection_second = matching.linear_assignment(dists, thresh=0.5)
        for itracked, idet in matches:
            track = r_tracked_stracks[itracked]
            det = detections_second[idet]
            if track.state == TrackState.Tracked:
                track.update(det, self.frame_id)
                activated_stracks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)

        for it in u_track:
            track = r_tracked_stracks[it]
            if not track.state == TrackState.Lost:
                track.mark_lost()
                lost_stracks.append(track)

        '''Deal with unconfirmed tracks, usually tracks with only one beginning frame'''
        detections = [detections[i] for i in u_detection]
        dists = matching.iou_distance(unconfirmed, detections)
        if not self.args.mot20:
            dists = matching.fuse_score(dists, detections)
        matches, u_unconfirmed, u_detection = matching.linear_assignment(dists, thresh=0.7)
        for itracked, idet in matches:
            unconfirmed[itracked].update(detections[idet], self.frame_id)
            activated_stracks.append(unconfirmed[itracked])
        for it in u_unconfirmed:
            track = unconfirmed[it]
            track.mark_removed()
            removed_stracks.append(track)

        ''' Step 4: Init new stracks '''
        for inew in u_detection:
            track = detections[inew]
            if track.score < self.new_track_thresh:
                continue

            track.activate(self.kalman_filter, self.frame_id)
            activated_stracks.append(track)

        ''' Step 5: Update state '''
        for track in self.lost_stracks:
            if self.frame_id - track.end_frame > self.max_time_lost:
                track.mark_removed()
                removed_stracks.append(track)

        ''' Merge '''
        self.tracked_stracks = [t for t in self.tracked_stracks if t.state == TrackState.Tracked]
        self.tracked_stracks = joint_stracks(self.frame_id, self.tracked_stracks, activated_stracks, 'call from 2')
        self.tracked_stracks = joint_stracks(self.frame_id, self.tracked_stracks, refind_stracks, 'call from 3')
        self.lost_stracks = sub_stracks(self.lost_stracks, self.tracked_stracks)
        self.lost_stracks.extend(lost_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.removed_stracks)
        self.removed_stracks.extend(removed_stracks)
        self.tracked_stracks, self.lost_stracks = remove_duplicate_stracks(self.tracked_stracks, self.lost_stracks)

        '''Example
        (for tenth frame)
        OT_1_(1-10)
        OT_2_(1-10)
        ==================
        repr: 'OT_{}_({}-{})'.format(self.track_id, self.start_frame, self.end_frame)
        '''
        # output_stracks = [track for track in self.tracked_stracks if track.is_activated]
        output_stracks = [track for track in self.tracked_stracks]
        

        return output_stracks

        """

# merge two strack lists
def joint_stracks(frame_id, tlista, tlistb, s):
    # print(s)
    exists = {}
    res = {}
    for t in tlista:
        # existing track id
        exists[t.track_id] = 1
        res[t.track_id] = t
    for t in tlistb:
        tid = t.track_id
        # existing object
        if tid in exists:
            # merge new kpts to existing track
            # print('t.kpts', t.kpts, sep='\n')
            # print('res[tid].kpts')
            # print(res[tid].kpts)
            # print('t.kpts')
            # print(t.kpts)
            res[tid].update_kpts(t.kpts)
        # new object
        else:
            exists[tid] = 1
            res[tid] = t
    # print('res.values()')
    # print(*[t.kpts for t in res.values()])
    # if frame_id == 4: exit(-1)
    return list(res.values())


def sub_stracks(tlista, tlistb):
    stracks = {}
    for t in tlista:
        stracks[t.track_id] = t
    for t in tlistb:
        tid = t.track_id
        if stracks.get(tid, 0):
            del stracks[tid]
    return list(stracks.values())


def remove_duplicate_stracks(stracksa, stracksb):
    pdist = matching.iou_distance(stracksa, stracksb)
    pairs = np.where(pdist < 0.15)
    dupa, dupb = list(), list()
    for p, q in zip(*pairs):
        timep = stracksa[p].frame_id - stracksa[p].start_frame
        timeq = stracksb[q].frame_id - stracksb[q].start_frame
        if timep > timeq:
            dupb.append(q)
        else:
            dupa.append(p)
    resa = [t for i, t in enumerate(stracksa) if not i in dupa]
    resb = [t for i, t in enumerate(stracksb) if not i in dupb]
    return resa, resb
