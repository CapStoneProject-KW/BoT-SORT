if __name__ == '__main__':
    from server.model import run_model
    from server.scoring import run_scoring
    import pprint

    mode = 'tracking' # 'detection' or 'tracking'
    data_path_answer = './datasets/last_dance/antifragile/answer.mp4'
    data_path_user = './datasets/last_dance/antifragile/user_sync.mp4' # Path to video
    
    if mode == 'detection':
        det_image_answer, det_result_answer = run_model(mode, data_path_answer)
        # print(type(det_image_answer), det_image_answer)
        det_image_user, det_result_user = run_model(mode, data_path_user)
        # print(type(det_image_user), det_image_user)
        print('[Detection Result]')
        print('Answer Video')
        pprint.pprint(det_result_answer)
        print('User Video')
        pprint.pprint(det_result_user)
    elif mode == 'tracking':
        kpt_result_answer, mot_result_answer = run_model(mode, data_path_answer)
        kpt_result_user, mot_result_user = run_model(mode, data_path_user)
        print('[Keypoint Result]')
        print('Answer Video')
        pprint.pprint(kpt_result_answer)
        print('User Video')
        pprint.pprint(kpt_result_user)
        print('[Tracking Result]')
        print('Answer Video')
        # pprint.pprint(mot_result_answer)
        print(mot_result_answer[10])
        print(mot_result_answer[100])
        print('User Video')
        # pprint.pprint(mot_result_user)
        print(mot_result_user[10])
        print(mot_result_user[100])

        matches = [[1, 1], [2, 2], [3, 3], [4, 4], [5, 5]]
        pose_scores = run_scoring('pose', 
                                kpt_result_user, 
                                kpt_result_answer, 
                                matches=matches,
                                distance='euclidean', 
                                score='simple')
        movement_scores = run_scoring('movement', 
                                mot_result_user, 
                                mot_result_answer, 
                                matches=matches,
                                distance='euclidean', 
                                score='simple')
        print('[Pose Score]')
        for track_id, score in pose_scores.items():
            print(f"ID: {track_id} Score: {score}")
        print('[Movement Score]')
        for track_id, score in movement_scores.items():
            print(f"ID: {track_id} Score: {score}")
    