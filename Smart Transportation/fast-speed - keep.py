import cv2
from ultralytics import YOLO
from collections import defaultdict
import numpy as np

VIDEO_PATH = r'E:\Autonomous driving\ultralytics-main\deteciton\video\抖音2024918-824647.mp4'
RESULT_PATH = r'E:\Autonomous driving\ultralytics-main\deteciton\video\抖音2024918-824647-speed-fast.mp4'
model = YOLO("yolov8n.pt")
track_history = defaultdict(lambda: [])
OBJ_list = ["person"]
SPEED_THRESHOLD = 4.0  # 速度阈值（单位：米/秒）

def calculate_speed(prev_pos, curr_pos, frame_rate):
    """
    Calculate the speed of an object based on its positions in two consecutive frames.
    """
    distance_pixels = np.linalg.norm(curr_pos - prev_pos)
    distance_meters = distance_pixels / 100  # Assuming 1 pixel = 0.01 meters
    time_seconds = 1 / frame_rate
    speed_mps = distance_meters / time_seconds
    return speed_mps

if __name__ == '__main__':
    # 打开视频文件
    capture = cv2.VideoCapture(VIDEO_PATH)
    if not capture.isOpened():
        print('Error opening video file.')
        exit()

    # 获取视频的一些基本属性
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))  # 宽度
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))  # 高度
    fps = capture.get(cv2.CAP_PROP_FPS)  # 帧率
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))  # 帧数

    print(f'视频宽度: {width}')
    print(f'视频高度: {height}')
    print(f'视频帧率: {fps}')
    print(f'视频总帧数: {frame_count}')

    # 初始化视频写入器
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(RESULT_PATH, fourcc, fps, (width, height))

    # 逐帧读取视频
    prev_positions = defaultdict(lambda: np.array([0, 0]))  # 记录每个跟踪对象上一帧的位置
    track_colors = defaultdict(lambda: (0, 255, 0))  # 默认颜色为绿色
    frame_index = 0  # 当前帧索引

    while capture.isOpened():
        ret, frame = capture.read()
        if not ret:
            break

        results = model.track(frame, persist=True)
        a_frame = results[0].plot()
        boxes = results[0].boxes.xywh.cpu()
        track_ids = results[0].boxes.id.int().cpu().tolist()

        for idx, (box, track_id) in enumerate(zip(boxes, track_ids)):
            x, y, w, h = box
            track = track_history[track_id]
            curr_pos = np.array([x, y])
            track.append((float(x), float(y)))

            # 如果历史轨迹超过一定长度，删除最老的位置
            if len(track) > 50:
                track.pop(0)

            # 计算速度
            speed = calculate_speed(prev_positions[track_id], curr_pos, fps)

            # 更新当前位置
            prev_positions[track_id] = curr_pos

            # 显示速度
            text = f"Speed: {speed:.2f} m/s"
            cv2.putText(a_frame, text, (int(x) - 50, int(y) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            # 一旦超速，颜色变为红色，并且不再变回绿色
            if speed > SPEED_THRESHOLD and track_colors[track_id] == (0, 255, 0):
                track_colors[track_id] = (0, 0, 255)

            # 根据记录的颜色绘制框
            color = track_colors[track_id]
            x1, y1, x2, y2 = int(x - w / 2), int(y - h / 2), int(x + w / 2), int(y + h / 2)
            cv2.rectangle(a_frame, (x1, y1), (x2, y2), color, 2)

        # 可视化
        cv2.imshow('yolo track', a_frame)
        cv2.waitKey(1)

        # 写入结果视频
        out.write(a_frame)

        # 按下 'q' 键退出循环
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 释放资源
    capture.release()
    out.release()
    cv2.destroyAllWindows()