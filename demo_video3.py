if __name__ == '__main__':
    from server.model import run_model
    from server.scoring import run_scoring
    import pprint
    import json
    import cv2
    from PIL import ImageFont, ImageDraw, Image
    import numpy as np

    def myPutText(src, text, pos, font_size, font_color) :
        img_pil = Image.fromarray(src)
        draw = ImageDraw.Draw(img_pil)
        font = ImageFont.truetype('fonts/font.ttf', font_size)
        # font = ImageFont.load_default()
        draw.text(pos, text, font=font, fill= font_color)
        return np.array(img_pil)

    # Load the videos
    # video1_path = './datasets/last_dance/antifragile/answer.mp4'
    # video2_path = './datasets/last_dance/antifragile/user_sync.mp4'
    # video1_path = './datasets/last_dance/single_person/youtube_video.mp4'
    # video2_path = './datasets/last_dance/single_person/user_video.mp4'
    video1_path = './runs/230525/exp/youtube_video.mp4'
    video2_path = './runs/230525/exp2/user_video.mp4'
    
    # 파일 경로
    output_video_path = 'output.mp4'
    move_score_file_path = 'runs/scoring_move.json'
    pose_score_file_path = 'runs/scoring_pose.json'

    with open(move_score_file_path, 'r') as f:
        move_scores = json.load(f)
    with open(pose_score_file_path, 'r') as f:
        pose_scores = json.load(f)

    # 비디오 캡처 객체 생성
    cap1 = cv2.VideoCapture(video1_path)
    cap2 = cv2.VideoCapture(video2_path)

    # 두 비디오의 프레임 크기, FPS, 코덱 정보 가져오기
    width1 = int(cap1.get(cv2.CAP_PROP_FRAME_WIDTH))
    height1 = int(cap1.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps1 = int(cap1.get(cv2.CAP_PROP_FPS))
    # fourcc1 = int(cap1.get(cv2.CAP_PROP_FOURCC))
    
    width2 = int(cap2.get(cv2.CAP_PROP_FRAME_WIDTH))
    height2 = int(cap2.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps2 = int(cap2.get(cv2.CAP_PROP_FPS))
    # fourcc2 = int(cap2.get(cv2.CAP_PROP_FOURCC))
    print(fps1, fps2)
    # 결과 영상의 너비, 높이, FPS, 코덱 설정
    output_width = width1 + width2
    output_height = min(height1, height2)
    output_fps = min(fps1, fps2)
    # output_fourcc = fourcc1

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    output_cap = cv2.VideoWriter(output_video_path, fourcc, output_fps, (output_width, output_height))

    frame_number = 0
    while True:
        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read()

        # 두 영상 중 하나가 끝났을 경우 종료
        if not ret1 or not ret2:
            break

        # 두 영상의 높이를 동일하게 조정
        frame1 = cv2.resize(frame1, (width1, output_height))
        frame2 = cv2.resize(frame2, (width2, output_height))

        # 두 영상을 수평방향으로 이어 붙이기
        output_frame = cv2.hconcat([frame1, frame2])

        # 네모 박스 그리기
        box_width = 120
        box_height= 60
        box_x = (output_width - box_width) // 2
        box_y = 10
        cv2.rectangle(output_frame, (box_x, box_y), (box_x + box_width, box_y + box_height), (255,255,255), -1)

        # 텍스트 추가
        current_second = frame_number // output_fps
        # score = scores["1"].get(str(current_second), 0)
        # score = scores["1"].get(str(frame_number), '')

        move_score = move_scores["1"].get(str(current_second), 0)
        pose_score = pose_scores["1"].get(str(current_second), 0)
        move_score *= 100
        pose_score *= 100

        # Move 텍스트
        text_x = box_x + 10
        text_y = box_y + 10
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.8
        font_color = (0, 0, 0)
        font_thickness = 2
        font_size = 20
        # cv2.putText(output_frame, f'move: {int(move_score)}', (text_x, text_y), font, font_scale, font_color, font_thickness)
        output_frame = myPutText(output_frame, f'move: {int(move_score)}점', (text_x, text_y), font_size, font_color)

        # Pose 텍스트
        text_x = box_x + 10
        text_y = box_y + 30
        # cv2.putText(output_frame, f'pose: {int(pose_score)}', (text_x, text_y), font, font_scale, font_color, font_thickness)
        output_frame = myPutText(output_frame, f'pose: {int(pose_score)}점', (text_x, text_y), font_size, font_color)
        # 결과 영상에 프레임 저장
        output_cap.write(output_frame)

        frame_number += 1
        print(f'{frame_number}/1000')

    cap1.release()
    cap2.release()
    output_cap.release()