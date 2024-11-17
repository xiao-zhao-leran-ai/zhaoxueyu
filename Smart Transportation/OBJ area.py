import cv2
from ultralytics import YOLO
import numpy as np
from shapely.geometry import LineString

VIDEO_PATH = r'E:\Autonomous driving\ultralytics-main\deteciton\video\抖音2024918-824647.mp4'
RESULT_PATH = r'E:\Autonomous driving\ultralytics-main\deteciton\video\抖音2024918-824647-area.mp4'

model = YOLO("yolov8n.pt")

# 自定义一个 ObjectCounter 类来处理对象计数
class ObjectCounter:
    def __init__(self, view_img=True, reg_pts=None, names=None, draw_tracks=True, line_thickness=2):
        self.view_img = view_img
        self.reg_pts = reg_pts
        self.names = names
        self.draw_tracks = draw_tracks
        self.line_thickness = line_thickness

        # 检查提供的区域点是否合法
        if self.reg_pts is not None and len(self.reg_pts) >= 3:
            try:
                self.counting_region = LineString(self.reg_pts)
            except ValueError:
                print("Provided region points do not form a valid polygon.")
                raise
        else:
            print("Invalid Region points provided, region_points must be 2 for lines or >= 3 for polygons.")
            raise ValueError("Invalid Region points provided.")

    def start_counting(self, frame, results):
        # 在这里可以添加计数逻辑
        return frame

if __name__ == '__main__':
    # 打开视频文件
    capture = cv2.VideoCapture(VIDEO_PATH)
    assert capture.isOpened(), "Error reading video file"

    w, h, fps, F_count = (int(capture.get(x)) for x in (
        cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS, cv2.CAP_PROP_FRAME_COUNT))

    line_points = [(0, int(7 * h / 8)), (int(w), int(7 * h / 8))]
    # 确保 poly_points 至少包含三个顶点
    poly_points = [
        (int(w / 2), int(11 * h / 16)),
        (int(5* w / 8), int(11* h / 16)),
        (int(5 * w / 8), int(6 * h / 8)),
        (int(w / 2), int(6 * h / 8))
    ]
    # 将 poly_points 转换为正确的格式
    poly_points = np.array([poly_points], dtype=np.int32)

    # 初始化视频写入器
    video_writer = cv2.VideoWriter(RESULT_PATH, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))

    # 声明对象计数函数
    counter = ObjectCounter(
        view_img=True,  # 是否显示计数信息
        reg_pts=poly_points,  # 多边形区域
        names=model.names,
        draw_tracks=True,  # 轨迹
        line_thickness=2
    )

    # 逐帧读取视频
    while capture.isOpened():
        ret, frame = capture.read()
        if not ret:
            break

        results = model.track(frame, persist=True, show=False)

 
        cv2.polylines(frame,poly_points,True,(0,0,255),3)
        mask = np.zeros_like(frame)
        cv2.fillPoly(frame, poly_points, (0, 0, 255))
    
        
        frame = counter.start_counting(frame, results)
        frame = cv2.addWeighted(frame, 0.7, mask, 0.3, 0)
        video_writer.write(frame)
        
        # 按下 'q' 键退出循环
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 释放资源
    capture.release()
    video_writer.release()
    cv2.destroyAllWindows()