import cv2
from ultralytics import YOLO
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from PIL import ImageEnhance
import pytesseract

# 设置 Tesseract 路径
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# 视频输入和输出路径
VIDEO_PATH = r'E:\Autonomous driving\ultralytics-main\deteciton\video\m.mp4'
RESULT_PATH = r'E:\Autonomous driving\ultralytics-main\deteciton\video\m-code.mp4'

# 加载YOLO模型
lincense_plate_detector = YOLO(r"E:\CCPD2020\CCPD2020\ccpd_green\runs\detect\train\weights\best.pt")
coco_model = YOLO('yolov8n.pt')

# 打开视频文件
cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    print(f"Failed to open the video file at '{VIDEO_PATH}'. Please check the path and try again.")
    exit(1)

# 获取视频的基本属性
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

# 初始化视频写入器
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(RESULT_PATH, fourcc, fps, (width, height))

# 字体文件路径
font_path = r'E:\CCPD2020\CCPD2020\ccpd_green\Arial.ttf'
font_size = 30
font = ImageFont.truetype(font_path, font_size)

# 车辆类别索引
vehicles = [2, 3, 5, 7]

# 主循环
ret = True
while ret:
    ret, frame = cap.read()
    if not ret:
        break

    # 使用YOLOv8进行车辆检测
    results = coco_model(frame)
    detections = results[0]
    boxes = detections.boxes.xyxy.cpu().numpy()
    labels = detections.boxes.cls.cpu().numpy()
    confidences = detections.boxes.conf.cpu().numpy()
    
    # 遍历每个检测到的车辆
    for i, box in enumerate(boxes):
        if int(labels[i]) in vehicles:
            x1, y1, x2, y2 = box.astype(int)
            
            # 裁剪目标区域
            cropped_region = frame[y1:y2, x1:x2]
            
            # 对裁剪后的目标区域进行车牌检测
            cropped_results = lincense_plate_detector(cropped_region)
            cropped_boxes = cropped_results[0].boxes.xyxy.cpu().numpy()
            
            # 遍历每个检测到的车牌
            for j, cbox in enumerate(cropped_boxes):
                cx1, cy1, cx2, cy2 = cbox.astype(int)
                
                # 裁剪车牌区域
                license_plate_region = cropped_region[cy1:cy2, cx1:cx2]
                
                # 图像增强
 
                # enhancer = ImageEnhance.Contrast(pil_image)
                # enhanced_image = enhancer.enhance(2)
        
                # 获取当前图像的尺寸
                current_height, current_width ,_= license_plate_region.shape

                # 定义新的尺寸（例如放大六倍）
                new_width, new_height = current_width * 6, current_height * 6

                # 使用 OpenCV 的 resize 方法放大图像
                resized_image = cv2.resize(license_plate_region, (new_width, new_height), interpolation=cv2.INTER_CUBIC)

                # 确保 resized_image 是 NumPy 数组
                pil_imag = Image.fromarray(resized_image)
                            # 转换为灰度图像
                gray_image = np.array(pil_imag.convert('L'))
                
                
                # 进行二值化处理
                _, binary_img = cv2.threshold(gray_image, 127, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
                    # _, binary = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
                # pil_image = Image.fromarray(binary_img)
                # gray_image = Image.fromarray(gray_image)
                # gray_image.show ()
                # 使用 Tesseract 提取文字
                config = '--oem 3 --psm 6'
                text = pytesseract.image_to_string(binary_img, lang='chi_sim+eng', config=config).strip()
                
                # 在图像上绘制识别的文字
                if text:
                    # 获取文字的尺寸
                    text_width, text_height = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
                    text_width += 20  # 添加边缘
                    text_height += 20
                    
                    # 创建一个白色背景的图像
                    text_img = Image.new('RGB', (text_width, text_height), color=(255, 255, 255))
                    draw = ImageDraw.Draw(text_img)
                    
                    # 在图像上绘制文本
                    draw.text((10, 10), text, fill=(0, 0, 0), font=font)
                    
                    # 将 PIL 图像转换回 OpenCV 图像
                    text_cv = cv2.cvtColor(np.array(text_img), cv2.COLOR_RGB2BGR)
                    
                    # 确定覆盖的位置（左上角坐标）
                    position = (x1, y1 - text_height)
                    
                    # 获取 text_cv 的尺寸
                    text_cv_height, text_cv_width = text_cv.shape[:2]
                    
                    # 确保 text_cv 的尺寸与 frame 中预留的区域尺寸一致
                    if text_cv_height > y1 or text_cv_width > (x2 - x1):
                        continue  # 如果尺寸不合适，跳过这次绘制
                        
                    # 将识别结果绘制到视频帧上
                    frame[y1 - text_height:y1, x1:x1 + text_cv_width] = text_cv[:text_cv_height, :]
                    
                    # 绘制车牌框
                    cv2.rectangle(frame, (x1 + cx1, y1 + cy1), (x1 + cx2, y1 + cy2), (0, 0, 255), 2)
    
    # 写入帧
    out.write(frame)
    
    # 显示当前帧
    cv2.imshow('Frame', frame)
    
    # 按下 'q' 键退出循环
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 释放资源
cap.release()
out.release()
cv2.destroyAllWindows()