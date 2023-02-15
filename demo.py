if __name__ == '__main__':
    from server.model import run_model
    from server.scoring import run_scoring

    mode = 'tracking' # 'detection' or 'tracking'
    data_path_answer = './datasets/last_dance/single_person/youtube_video.mp4'
    data_path_user = './datasets/last_dance/single_person/user_video.mp4' # Path to video
    ckpt_path = './pretrained/yolov7.pt' if mode == 'detection' \
                    else './pretrained/yolov7-w6-pose.pt' # Path to model weight file
    
    if mode == 'detection':
        det_result_answer = run_model(mode, data_path_user, ckpt_path)
        det_result_user = run_model(mode, data_path_user, ckpt_path)
        print('[Detection Result]')
        print('Answer Video')
        print(det_result_user)
        print('User Video')
        print(det_result_answer)
    elif mode == 'tracking':
        kpt_result_answer, mot_result_answer = run_model(mode, 
                                                    data_path_answer, 
                                                    ckpt_path)
        kpt_result_user, mot_result_user = run_model(mode, 
                                                    data_path_user, 
                                                    ckpt_path)
        print('[Keypoint Result]')
        print('Answer Video')
        print(kpt_result_answer)
        print('User Video')
        print(kpt_result_user)
        print('[Tracking Result]')
        print('Answer Video')
        print(mot_result_answer)
        print('User Video')
        print(mot_result_user)

        pose_scores = run_scoring('pose', 
                                kpt_result_answer, 
                                kpt_result_user, 
                                distance='weighted', 
                                score='simple')
        movement_scores = run_scoring('movement', 
                                kpt_result_answer, 
                                kpt_result_user, 
                                distance='weighted', 
                                score='simple')
        print('[Pose Score]')
        for track_id, score in pose_scores.items():
            print(f"ID: {track_id} Score: {score}")
        print('[Movement Score]')
        for track_id, score in movement_scores.items():
            print(f"ID: {track_id} Score: {score}")
    