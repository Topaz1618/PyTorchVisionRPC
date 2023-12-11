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
from task_utils import update_task_info, get_task_info
from log_handler import logger
from extensions import DetectionTaskManager


def pre_process(task_id, images_path, preprocess_dir):
    res_dict = dict()

    file_list = os.listdir(images_path)
    file_count = len(file_list)
    logger.info(f"Start Preprocessing ...")
    update_task_info(task_id, TaskInfoKey.LOG.value, f"Start Preprocessing ...")
    update_task_info(task_id, TaskInfoKey.TOTAL_FILE_COUNT.value, file_count)

    for idx, filename in enumerate(file_list):
        file_extension = filename.split(".")[-1]
        file_name = filename.split(".")[0]
        res_dict[filename] = {"file_type": file_extension.upper()}

        detection_type_list = ["pdf", "png", "jpg", "jpeg"]
        img_extension_list = ["png", "jpg", "jpeg"]
        if file_extension not in detection_type_list:
            continue

        logger.info(f"Preprocessing progress: {idx}/{file_count} ")
        update_task_info(task_id, TaskInfoKey.LOG.value, f"Preprocessing progress: {idx}/{file_count}")
        if file_extension in img_extension_list:
            shutil.copy(os.path.join(images_path, filename), os.path.join(preprocess_dir, f"{file_name}_page_0.png"))
        else:

            with pdfplumber.open(os.path.join(images_path, filename)) as pdf:
                for page_num, page in enumerate(pdf.pages):
                    img = page.to_image()
                    pdf_image = page.to_image().original

                    open_cv_image = cv2.cvtColor(np.array(pdf_image), cv2.COLOR_RGB2BGR)
                    cv2.imwrite(os.path.join(preprocess_dir, f"{file_name}_page_{page_num}.png"), open_cv_image)

    return res_dict


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
        print(img_name, len(result), len(result[0]))

        for line_num, line in enumerate(result[0]):

            location = line[0]
            text, confidence = line[1]
            #             print(f"文件: {img_name} 文本：{text} 置信度：{confidence} 坐标: {location}")

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
    if not os.path.exists(preprocess_dir):
        os.mkdir(preprocess_dir)

    # 预测图像类别
    res_dict = pre_process(task_id, images_path, preprocess_dir)
    res = predict_image_class(task_id, images_path, preprocess_dir, ocr_model, res_dict)

    update_task_info(task_id, TaskInfoKey.RESULT.value, res)
    update_task_info(task_id, TaskInfoKey.LOG.value, f"Detection Task: [{task_id}] Already Completed!")
    update_task_info(task_id, "task_status", TaskStatus.COMPLETED.value)


    return True


if __name__ == "__main__":
    detect_floder = "detect_demo1"
    task_id = "123"
    node = "worker1"
    handler(detect_floder, task_id, node)



