
from yolo import YOLO
from yolo import detect_video



if __name__ == '__main__':
    video_path='../test_video.mp4'
    detect_video(YOLO(), video_path)
