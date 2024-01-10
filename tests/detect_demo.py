import argparse
import sys
from time import sleep, time
from enum import Enum
import cv2

# 创建参数解析器
parser = argparse.ArgumentParser(description='Description of your script')

# 添加需要的参数
parser.add_argument('--task_id', type=str, help='Id of detect task')
# 添加需要的参数
parser.add_argument('--model', type=str, help='Id of detect task')

# 解析命令行参数
args = parser.parse_args()


class DetectionAlgorithm(Enum):
    YOLO = "YOLO"
    MaskRCNN = "Mask R-CNN"
    Tesseract = "Tesseract"


def task_demo(task_id):
    idx = 0
    while True:
        print(f"Detect task: {task_id} idx: {idx}")
        sleep(1)
        if idx == 3:
            idx = idx + "1"

        if idx == 10:
            break

        idx += 1

    print("done")
    return


if __name__ == "__main__":
    if not args.task_id:
        raise ValueError("please pass task_id")

    if not args.model:
        raise ValueError("Please pass model")

    model = args.model
    if model not in [algorithm.value for algorithm in DetectionAlgorithm]:
        raise TypeError("Valid detection algorithm.")

    task_id = args.task_id

    # 在这里可以使用参数值执行相应的逻辑
    print(f"task_id: {task_id}")

    res = task_demo(task_id)
    if res:
        sys.exit()