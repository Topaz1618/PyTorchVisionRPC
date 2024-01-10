import os
import sys
import shutil

import cv2
import torch
import numpy as np
import pdfplumber
import paddle
from paddleocr import PaddleOCR, draw_ocr

parent_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(parent_dir)

from enums import FileFormatType, MaterialType, MaterialTitleType, TaskStatus, TaskInfoKey
from task_utils import update_task_info, get_task_info, pre_process
from log_handler import logger
from extensions import DetectionTaskManager


def predict_image_class(task_id, images_path, preprocess_dir, ocr_model, res_dict):
    pre_process_files = os.listdir(preprocess_dir)
    pre_process_files_count = len(pre_process_files)
    result_path = os.path.join("output", "detect", task_id)

    colors = np.random.randint(125, 255, (80, 3))

    files_count = len(pre_process_files)
    update_task_info(task_id, TaskInfoKey.PROCESSED_FILE_COUNT.value, files_count)

    for idx, img_name in enumerate(pre_process_files):
        logger.info(f"Task_id:{task_id} Preprocessing FILE: {img_name} current progress: {idx}/{files_count} ")
        update_task_info(task_id, TaskInfoKey.LOG.value,
                         f" Task_id:{task_id} Preprocessing FILE: {img_name} Current Progress {idx}/{files_count}")

        file_extension = img_name.split(".")[-1]

        if file_extension != "png":
            continue

        file_name = img_name.split("_page")[0]

        matching_keys = [key for key in res_dict.keys() if file_name in key]
        if not matching_keys:
            continue

        related_pdf_name = matching_keys[0]

        result = ocr_model.ocr(os.path.join(preprocess_dir, img_name))

        if not result:
            continue

        if not result[0]:
            continue

        for line_num, line in enumerate(result[0]):

            location = line[0]
            text, confidence = line[1]

            detection_res = {
                "text": text,
                "bbox": location,
                "score": str(confidence),
            }

            if not res_dict[related_pdf_name].get("result"):
                res_dict[related_pdf_name]["result"] = list()

            res_dict[related_pdf_name]["result"].append(detection_res)
            logger.info(
                f'Task: {task_id} The file {img_name} predicted line: {line_num} / {len(result[0])}')
            update_task_info(task_id, TaskInfoKey.LOG.value,
                             f'Task: {task_id} The file {img_name} predicted line: {line_num} / {len(result[0])}')

        update_task_info(task_id, TaskInfoKey.LOG.value, f'Task: {task_id} The file {img_name} predicted done.')
        update_task_info(task_id, TaskInfoKey.PROCESSED_FILE_COUNT.value, idx + 1)
        logger.info(f'Task: {task_id} The file {img_name} predicted done.')

    return res_dict


def handler(detect_floder, task_id, node):
    ocr_model = PaddleOCR(use_gpu=False)
    logger.info("model loaded")
    update_task_info(task_id, TaskInfoKey.LOG.value, f"model loaded")

    images_path = detect_floder
    #     images_path = os.path.join("temp_storage", detect_floder)
    preprocess_dir = os.path.join(images_path, "pre_process")

    res_dict = pre_process(task_id, images_path, preprocess_dir)
    res = predict_image_class(task_id, images_path, preprocess_dir, ocr_model, res_dict)

    update_task_info(task_id, TaskInfoKey.RESULT.value, res)
    update_task_info(task_id, TaskInfoKey.LOG.value, f"Detection Task: [{task_id}] Already Completed!")
    update_task_info(task_id, "task_status", TaskStatus.COMPLETED.value)

    return True




