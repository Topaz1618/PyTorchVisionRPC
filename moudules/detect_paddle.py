import os
import sys
import shutil
import uuid

import pdfplumber
import pytesseract
import re
import docx
from enum import Enum
from time import sleep, time
from paddleocr import PaddleOCR, draw_ocr

parent_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(parent_dir)

from enums import FileFormatType, MaterialType, MaterialTitleType, TaskStatus, TaskInfoKey
from config import TEMP_STORAGE_DIR
from task_utils import update_task_info, get_task_info
from log_handler import logger

title = ["姓名", "性别", "民族", "出生", "住址", "公民身份号码"]


def detect_img(task_id, detect_folder_path):
    file_dic = {}
    file_list = os.listdir(detect_folder_path)
    total_file_count = len(file_list)
    update_task_info(task_id, TaskInfoKey.TOTAL_FILE_COUNT.value, total_file_count)
    processed_file_count = 0

    ocr = PaddleOCR(use_gpu=False)
    result = ocr.ocr(img_path)
    for file_name in file_list:
        update_task_info(task_id, TaskInfoKey.LOG.value, f"{task_id} Processing file: {file_name}")
        file_path = os.path.join(detect_folder_path, file_name)

        res_dict = {}
        current_title = None
        for line in result[0]:
            # 提取信息
            # print(f"{line} \n")

            location = line[0]
            text, confidence = line[1]

            for t in title:
                if t in text:
                    current_title = t
                    text = text.replace(t, '').strip()  # 移除标题，留下内容
                    break
            if current_title:
                if current_title not in res_dict:
                    res_dict[current_title] = text
                else:
                    res_dict[current_title] += ' ' + text

            print(f"文本：{text} 置信度：{confidence}")

        print(res_dict)
    return res_dict


if __name__ == "__main__":
    task_id = "4ba5ff7d-f43c-4676-9dd8-57dc4339f1d6"
    img_path = 'data/0004.jpg'
    detect_img(task_id, img_path)