import cv2
from ultralytics import YOLO,solutions
import numpy as np
VIDEO_PATH = r'E:\Autonomous driving\ultralytics-main\deteciton\video\抖音2024918-824647.mp4'
RESULT_PATH = r'E:\Autonomous driving\ultralytics-main\deteciton\video\抖音2024918-824647-area.mp4'

model=YOLO("yolov8n.pt")

if __name__ == '__main__':
    # 打开视频文件
    capture = cv2.VideoCapture(VIDEO_PATH)
    assert capture.isOpened(), "Error readinng video file"


    w,h,fps,F_count=(int(capture.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH,cv2.CAP_PROP_FRAME_HEIGHT,cv2.CAP_PROP_FPS,cv2.CAP_PROP_FRAME_COUNT))
    
    line_points=[(0,int(7*h/8)),(int(w),int(7*h/8))  ]

    poly_points = [
        (int(w / 2), int(11 * h / 16)),
        (int(5* w / 8), int(11* h / 16)),
        (int(5 * w / 8), int(6 * h / 8)),
        (int(w / 2), int(6 * h / 8))]
    
    poly_points = np.array(poly_points, dtype=np.int32)
    # 初始化视频写入器

    video_writer=cv2.VideoWriter(RESULT_PATH,cv2.VideoWriter_fourcc(*'mp4v'),fps,(w,h))
    
    #声明对象计数函数
    counter=solutions.ObjectCounter(
        view_img=True,  #是否显示计数信息
        reg_pts=poly_points,
        names=model.names,
        draw_tracks=True,#轨迹
        line_thickness=2
        
    )
    # 逐帧读取视频
    while capture.isOpened():
        ret, frame = capture.read()
        if not ret:
            break
    
        results=model.track(frame,persist=True,show=False)
        frame=counter.start_counting(frame,results)

        mask=cv2.fillPoly(frame,[poly_points], (0, 0, 255))
        frame = cv2.addWeighted(frame, 0.9, mask, 0.1, 0)
        video_writer.write(frame)
        # 按下 'q' 键退出循环
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 释放资源
    capture.release()
    # out.release()
    video_writer.release
    cv2.destroyAllWindows()