import os
import logging
import datetime

# 获取当前日期
current_date = datetime.datetime.now().strftime("%Y-%m-%d")

# 配置日志记录器
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

# 创建一个文件处理器，将日志写入带有日期的文件
log_file = os.path.join("logfiles", f"log_file_{current_date}.log")
file_handler = logging.FileHandler(log_file)
file_handler.setLevel(logging.INFO)

# 创建一个控制台处理器，将日志输出到屏幕
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)

formatter = logging.Formatter('%(asctime)s [%(levelname)s] (%(filename)s:%(lineno)d) %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

# 将格式器添加到处理器
file_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)

# 将处理器添加到日志记录器
logger.addHandler(file_handler)
logger.addHandler(console_handler)

# 示例日志输出
# logger.debug('Debug message')
# logger.info('Info message')
# logger.warning('Warning message')
# logger.error('Error message')
# logger.critical('Critical message')