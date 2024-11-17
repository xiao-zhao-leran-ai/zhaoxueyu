import logging
import os
from yaml import *
log_filename = os.path.join(DQN_log_path, "training.log")

# 配置日志
logging.basicConfig(
    filename=log_filename,
    filemode='a',  # 追加模式
    format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    level=logging.INFO
)

# 日志记录函数
def log_info(message):
    logging.info(message)