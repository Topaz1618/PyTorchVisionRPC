"""
For test
"""

import os
import sys
import argparse
from time import sleep, time

# parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# sys.path.append(parent_dir)


from task_utils import get_task_info, update_task_info
from compression_utils import decompress_zip
from enums import DetectionAlgorithm
from moudules.default_detect import pydetect_loop


# 创建参数解析器
parser = argparse.ArgumentParser(description='Description of your script')

# 添加需要的参数
parser.add_argument('--task_id', type=str, help='Id of detect task')
parser.add_argument('--model', type=str, default='Tesseract', choices=['Tesseract', ], help='Detection algorithm')

# 解析命令行参数
args = parser.parse_args()


def main():
    # if not args.task_id:
    #     raise ValueError("no task id")

    # Todo: 从

    model = args.model
    print(args.model)

    task_id = '4ba5ff7d-f43c-4676-9dd8-57dc4339f1d6'

    task_info = get_task_info(task_id)
    print(task_info)
    """
    Todo: 
        Redis detect_file = task_info.get("detect_file") 
        mongoDB 下载文件 detect_file 到 temp_storage
        (测试使用本地文件)
    """

    update_task_info(task_id, "log", "开始解压 (Unzipping)...")
    zip_name = task_info.get("detect_file")
    uncompress_file_path = decompress_zip(task_id, os.path.join("temp_storage", zip_name))

    if model == DetectionAlgorithm.TE.value:
        res = pydetect_loop(task_id, uncompress_file_path)

    if res:
        print("释放 worker")


if __name__ == "__main__":
    # 指定PDF文件路径
    print(" === 文件识别任务调试用方法测试开始 ===")
    main()

    print(" === 文件识别任务调试用方法测试结束 ===")
