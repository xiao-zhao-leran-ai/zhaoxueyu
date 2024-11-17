import cv2  # 导入OpenCV库
from grabscreen import grab_screen  # 导入屏幕抓取模块

import pytesseract
from PIL import Image
import numpy as np


# 将 PIL 形式的图像转换为 OpenCV 形式的图像
def pil_to_cv(pil_image):
    img_array = np.array(pil_image)
    return cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

# 灰度化和二值化
def preprocess_image(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    scale_percent = 600  # 放大的百分比
    width = int(gray_image.shape[1] * scale_percent / 100)
    height = int(gray_image.shape[0] * scale_percent / 100)
    dim = (width, height)
    
    resized_image = cv2.resize(gray_image, dim, interpolation=cv2.INTER_LINEAR)
    binary_image = cv2.threshold(resized_image, 155, 255, cv2.THRESH_BINARY_INV)[1]
    return binary_image


# 识别数字
def recognize_digits(image):
    preprocessed_image = preprocess_image(image)
    text = pytesseract.image_to_string(preprocessed_image, lang='eng', config='-c tessedit_char_whitelist=0123456789')
    return text
