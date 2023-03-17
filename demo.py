if __name__ == '__main__':
    from server.model import run_model
    from server.scoring import run_scoring
    import pprint
    import json
    import cv2

    mode = 'tracking' # 'detection' or 'tracking' or 'show_video'
    # data_path_answer = './datasets/last_dance/single_person/youtube_video.mp4'
    # data_path_user = './datasets/last_dance/single_person/user_video.mp4' # Path to video
    data_path_answer = './datasets/last_dance/antifragile/answer.mp4'
    data_path_user = './datasets/last_dance/antifragile/user_sync.mp4' # Path to video

    if mode == 'detection':
        det_image_answer, det_result_answer = run_model(mode, data_path_answer)
        # print(type(det_image_answer), det_image_answer)
        det_image_user, det_result_user = run_model(mode, data_path_user)
        print(first_frame_img_path_answer)
        # print(type(det_image_user), det_image_user)
        print('[Detection Result]')
        print('Answer Video')
        pprint.pprint(det_result_answer)
        print('User Video')
        pprint.pprint(det_result_user)
    elif mode == 'show_video':
        video1 = cv2.VideoCapture(data_path_answer)
        video2 = cv2.VideoCapture(data_path_user)

        width = int(video1.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(video1.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(video1.get(cv2.CAP_PROP_FPS))

        output_path = 'runs/output.mp4'
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        output = cv2.VideoWriter(output_path, fourcc, fps, (2 * width, height))

        # Loop through the frames of the videos
        while True:
            # Read the frames from the videos
            ret1, frame1 = video1.read()
            ret2, frame2 = video2.read()
            
            # Break the loop if either video ends
            if not ret1 or not ret2:
                break
            
            # Get the score for this frame
            score1 = 0.8  # replace with the actual score from the dictionary
            score2 = 0.6  # replace with the actual score from the dictionary
            
            # Add the score text to the frames
            text1 = f'Score: {score1:.2f}'
            text2 = f'Score: {score2:.2f}'
            cv2.putText(frame1, text1, (int(width/2), 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(frame2, text2, (int(width/2), 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            
            # Concatenate the frames horizontally
            output_frame = cv2.hconcat([frame1, frame2])
            
            # Write the output frame to the video file
            output.write(output_frame)
            
        # Release the resources
        video1.release()
        video2.release()
        output.release()


    elif mode == 'tracking':
        kpt_result_answer, mot_result_answer = run_model(mode, data_path_answer)
        kpt_result_user, mot_result_user = run_model(mode, data_path_user)
        print('[Keypoint Result]')
        print('Answer Video')
        # pprint.pprint(kpt_result_answer)
        print('User Video')
        # pprint.pprint(kpt_result_user)
        print('[Tracking Result]')
        print('Answer Video')
        # pprint.pprint(mot_result_answer)
        # print(mot_result_answer[10])
        # print(mot_result_answer[100])
        print('User Video')
        # pprint.pprint(mot_result_user)
        # print(mot_result_user[10])
        # print(mot_result_user[100])
    

        # tmp_user = './runs/230130/exp/mot_result.json'
        # tmp_answer = './runs/230130/exp3/mot_result.json'

        with open('./runs/230130/exp2/mot_result.json', 'r') as f:
            tmp_user = json.load(f)
        with open('./runs/230131/exp/mot_result.json', 'r') as f:
            tmp_answer = json.load(f)

        matches = [['1', '1'], ['2', '2'], ['3', '3'], ['4', '4'], ['5', '5']]
        pose_scores = run_scoring('pose', 
                                kpt_result_user, 
                                kpt_result_answer, 
                                matches=matches,
                                distance='euclidean', 
                                score='simple')
        movement_scores = run_scoring('movement', 
                                tmp_user, 
                                tmp_answer, 
                                matches=matches,
                                distance='euclidean', 
                                score='simple')
        print('[Pose Score]')
        for track_id, score in pose_scores.items():
            print(f"ID: {track_id} Score: {score}")
        print('[Movement Score]')
        for track_id, score in movement_scores.items():
            print(f"ID: {track_id} Score: {score}")
