import cv2
from ultralytics import YOLO
from pathlib import Path

# img
# model=YOLO()
# results=model()
# res=results[0].plot()
# cv2.show()
# cv2.waikey()
VIDEO_PATH = r"E:\Autonomous driving\2.laneDection\Ultra-Fast-Lane-Detection-master\output_video.mp4"
RESULT_PATH = r"E:\Autonomous driving\2.laneDection\Ultra-Fast-Lane-Detection-master\output_video-obb.mp4"

model = YOLO("yolov8n-obb.pt")
names = model.names

if __name__ == '__main__':
    # 打开视频文件
    capture = cv2.VideoCapture(VIDEO_PATH)
    assert capture.isOpened(), "Error reading video file"

    # 获取视频的基本属性
    w, h, fps, F_count = (int(capture.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS, cv2.CAP_PROP_FRAME_COUNT))
    
    # 初始化视频写入器
    video_writer = cv2.VideoWriter(RESULT_PATH, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
  
    # 逐帧读取视频
    while capture.isOpened():
        ret, frame = capture.read()

        if not ret:
            break

        # 使用模型进行姿势估计
        results = model(frame, verbose=False)
        a_frame=results[0].plot()
        cv2.imshow('Pose',a_frame)
        
        video_writer.write(a_frame)
        
        # 按下 'q' 键退出循环
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 释放资源
    capture.release()
    video_writer.release()
    cv2.destroyAllWindows()