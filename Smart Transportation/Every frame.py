import cv2
from ultralytics import YOLO,solutions
VIDEO_PATH = r'./deteciton/video/抖音2024918-824647.mp4'
RESULT_PATH = r'./deteciton/video/抖音2024918-824647-1.mp4'
model=model=YOLO("yolov8n.pt")
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

    # 创建一个窗口来显示视频帧
    cv2.namedWindow('Frame', cv2.WINDOW_NORMAL)

    # 初始化视频写入器
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(RESULT_PATH, fourcc, fps, (width, height))

    # 逐帧读取视频
    while capture.isOpened():
        ret, frame = capture.read()
        if not ret:
            break
        #画线
        # cv2.line(frame,(0,int(height/2)),(int(width),int(height/2)),(0,0,255),3)
        results=model.track(frame,persist=True,verbose=True)
        #画区域
        # cv2.polylines(frame,polygonPoints,True,(0,0,255),3)
        #填充区域
        # mask=np.zeros_like(frame)
        # cv2.fillPoly()
        #融合两张图片
        #frame=cv2.addWeighted(frame,0.7,mask,0.3,0)
        # 显示当前帧
        cv2.imshow('Frame', frame)
        

        # 写入结果视频
        out.write(frame)

        # 按下 'q' 键退出循环
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 释放资源
    capture.release()
    out.release()
    cv2.destroyAllWindows()