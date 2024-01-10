import os
import sys
import uuid
import random
from datetime import datetime

import cv2
import torch
import numpy as np

parent_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(parent_dir)
current_directory = os.getcwd()


from enums import MaterialContentType, MaterialType, MaterialTitleType, TaskStatus, TaskInfoKey, TrainModelType, TaskType
from task_utils import update_task_info, get_task_info, create_task, pre_process
from log_handler import logger


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

    model_weights_path = '../moudules/YOLO/ultralytics_yolov5_master/yolov5s.pt'
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = torch.hub.load('../moudules/YOLO/ultralytics_yolov5_master', 'yolov5s',
                           source='local')  # or yolov5m, yolov5l, yolov5x, custom
    model.eval()

    model = model.to(device)

    logger.info("model loaded")
    update_task_info(task_id, TaskInfoKey.LOG.value, f"model loaded")

    images_path = detect_floder
    #     images_path = os.path.join("temp_storage", detect_floder)
    preprocess_dir = os.path.join(images_path, "pre_process")

    res_dict = pre_process(task_id, images_path, preprocess_dir)
    res = predict_image_class(task_id, images_path, preprocess_dir, model, res_dict)

    update_task_info(task_id, TaskInfoKey.RESULT.value, res)
    update_task_info(task_id, TaskInfoKey.LOG.value, f"Detection Task: [{task_id}] Already Completed!")
    update_task_info(task_id, TaskInfoKey.STATUS.value, TaskStatus.COMPLETED.value)

    print(res)

    return True


if __name__ == "__main__":
    detect_floder = "detect_demo"
    task_id = str(uuid.uuid4())  # 生成唯一的任务ID
    create_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    model = TrainModelType.RESNET.value
    node = "worker1"
    dataset = "coco"

    task = create_task(task_id, TaskType.DETECT.value, create_time, model, 'detect_demo1.zip')
    handler(detect_floder, task_id, node)






