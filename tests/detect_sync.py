"""
For test
"""

import os
import argparse
from time import sleep, time


from task_utils import get_task_info, update_task_info, remove_used_capacity
from compression_utils import decompress_zip
from enum import Enum
from enums import DetectionAlgorithm, TaskType, TaskInfoKey, ModelSource
from moudules.default_detect import pydetect_loop
from moudules.detect_paddle import detect_img
from log_handler import logger


# 创建参数解析器
parser = argparse.ArgumentParser(description='Description of your script')

# 添加需要的参数
parser.add_argument('--task_id', type=str, help='Id of detect task')
parser.add_argument('--model', type=str, default='Tesseract', choices=['Tesseract', ], help='Detection algorithm')
parser.add_argument('--node', type=str, help='Id of detect task')


# 解析命令行参数
args = parser.parse_args()


class TaskStatus(Enum):
    PENDING = 1
    IN_PROGRESS = 2
    COMPLETED = 3
    FAILED = 4
    CANCELLED = 5
    RETRYING = 6
    TIMEOUT = 7


def main():
    args = parser.parse_args()

    if not args.task_id:
        raise ValueError("no task id")

    if not args.node:
        raise ValueError("no node ")

    task_id = args.task_id
    node = args.node
    model = args.model

    # from ipdb import set_trace; set_trace()
    if isinstance(task_id, bytes):
        task_id = task_id.decode()

    task_info = get_task_info(task_id)
    model_source = task_info.get("model_source")

    """
    Todo: 
        Redis detect_file = task_info.get("detect_file") 
        mongoDB 下载文件 detect_file 到 temp_storage
        (测试使用本地文件)
    """

    zip_name = task_info.get("detect_file")

    logger.info(f"Starting Unzipping: {zip_name}")
    update_task_info(task_id, TaskInfoKey.LOG.value, f"Starting Unzipping: {zip_name}")
    uncompress_file_path = decompress_zip(task_id, os.path.join("temp_storage", zip_name))

    if model == DetectionAlgorithm.TESSERACT.value:
        res = pydetect_loop(task_id, uncompress_file_path)

    elif model == DetectionAlgorithm.YOLO.value:
        """
        if ModelType.built_in.value:
            # Todo: 检查文件是否存在本地指定目录，不在就下载
            threading.Thread(update_task_info(upload_process)) 
        else:
              # Todo: 检查项目目录是否存在，下载项目, 并解压。
        """
        # Todo: 从 task_id 获取 Model 是系统自带的，还是用户上传的

        if model_source == ModelSource.CUSTOM.value:
            pass
        else:
            pass

    elif model == DetectionAlgorithm.MaskRCNN.value:
        res = None

    elif model == DetectionAlgorithm.RESNET.value:
        res = None

    elif model == DetectionAlgorithm.PADDLEOCR.value:
        if model_source == ModelSource.CUSTOM.value:
            # Todo: call detect.py and pass number
            res = None
        else:
            res = detect_img(task_id, uncompress_file_path)
    
    else:
        raise ValueError("not valid detection algorithm type")

    if res:
        logger.warning(f"Task {task_id} Result: {res}")

        remove_used_capacity(node, TaskType.DETECT.value)


if __name__ == "__main__":
    # 指定PDF文件路径
    print(" === 文件识别任务调试用方法测试开始 ===")

    main()

    print(" === 文件识别任务调试用方法测试结束 ===")