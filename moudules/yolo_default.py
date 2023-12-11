import os
import torch
import sys
import random
import shutil

import cv2
import torch
import pdfplumber

import numpy as np
from pathlib import Path

# from YOLO.ultralytics_yolov5_master.models.experimental import attempt_load
# from models.experimental import attempt_load


parent_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(parent_dir)
current_directory = os.getcwd()
print("当前工作目录：", current_directory)

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


def predict_image_class(task_id, images_path, preprocess_dir, model, res_dict):
    pre_process_files = os.listdir(preprocess_dir)
    pre_process_files_count = len(pre_process_files)
    result_path = os.path.join("output", "detect", task_id)

    colors = np.random.randint(125, 255, (80, 3))
    files_count = len(pre_process_files)
    update_task_info(task_id, TaskInfoKey.PROCESSED_FILE_COUNT.value, files_count)

    for idx, img_name in enumerate(pre_process_files):
        #         print(f"Task_id:{task_id} {idx}/{pre_process_files_count} img_name: {img_name}")
        logger.info(f"Task_id:{task_id} Preprocessing FILE: {img_name} current progress: {idx + 1}/{files_count} ")
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

        img = cv2.imread(os.path.join(preprocess_dir, img_name))

        results = model(img)

        color = colors[int(random.randint(1, 10))]

        detected_objects = results.pandas().xyxy[0]

        for index, obj in detected_objects.iterrows():
            x1, y1, x2, y2, conf, label = int(obj[0]), int(obj[1]), int(obj[2]), int(obj[3]), obj[4], int(obj[5])

            if conf < 0.5:
                continue

            cv2.rectangle(img, (x1, y1), (x2, y2), (int(color[0]), int(color[1]), int(color[2])), 2)
            label_text = f"{model.names[label]}: {conf:.2f}"
            cv2.putText(img, label_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (int(color[0]), int(color[1]), int(color[2])), 2)

            detection_res = {
                "class_id": int(label),
                "label": label_text,
                "bbox": [x1, y1, x2, y2],
                "score": conf,
            }

            if not res_dict[related_pdf_name].get("result"):
                res_dict[related_pdf_name]["result"] = list()

            res_dict[related_pdf_name]["result"].append(detection_res)
            cv2.imwrite(os.path.join(result_path, f'{file_name}.jpg'), img)

            update_task_info(task_id, TaskInfoKey.LOG.value,
                             f'Task: {task_id} The file {img_name} predicted current object class is: {label_text} axis: {x1} {y1} {x2} {y2}')
            update_task_info(task_id, TaskInfoKey.PROCESSED_FILE_COUNT.value, idx)

        update_task_info(task_id, TaskInfoKey.LOG.value, f'Task: {task_id} The file {img_name} predicted done.')
        update_task_info(task_id, TaskInfoKey.PROCESSED_FILE_COUNT.value, idx + 1)
        logger.info(f'Task: {task_id} The file {img_name} predicted done.')

    return res_dict


def handler(detect_floder, task_id, node):
    cache_dir = 'YOLO'
    torch.hub.set_dir(cache_dir)

    model_weights_path = 'moudules/YOLO/ultralytics_yolov5_master/yolov5s.pt'
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = torch.hub.load('moudules/YOLO/ultralytics_yolov5_master', 'yolov5s',
                           source='local')  # or yolov5m, yolov5l, yolov5x, custom
    model.eval()

    model = model.to(device)

    logger.info("model loaded")
    update_task_info(task_id, TaskInfoKey.LOG.value, f"model loaded")

    images_path = detect_floder
    #     images_path = os.path.join("temp_storage", detect_floder)
    preprocess_dir = os.path.join(images_path, "pre_process")
    if not os.path.exists(preprocess_dir):
        os.mkdir(preprocess_dir)

    res_dict = pre_process(task_id, images_path, preprocess_dir)
    res = predict_image_class(task_id, images_path, preprocess_dir, model, res_dict)

    update_task_info(task_id, TaskInfoKey.RESULT.value, res)
    update_task_info(task_id, TaskInfoKey.LOG.value, f"Detection Task: [{task_id}] Already Completed!")
    update_task_info(task_id, TaskInfoKey.STATUS.value, TaskStatus.COMPLETED.value)

    task_obj = DetectionTaskManager()
    task_obj.update_task(task_id, TaskInfoKey.RESULT.value, res)
    task_obj.update_task(task_id, "task_status", TaskStatus.COMPLETED.value)
    task_obj.close()

    return True


if __name__ == "__main__":
    detect_floder = "detect_demo1"
    task_id = "123"
    node = "worker1"
    handler(detect_floder, task_id, node)






