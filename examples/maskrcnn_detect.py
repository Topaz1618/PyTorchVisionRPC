import os
import sys
import uuid
import shutil

import cv2
import numpy as np
import pdfplumber
from datetime import datetime

path_to_frozen_inference_graph = '../moudules/Mask_RCNN/data/frozen_inference_graph_coco.pb'
path_coco_model = '../moudules/Mask_RCNN/data/mask_rcnn_inception_v2_coco_2018_01_28.pbtxt'
class_label = "../moudules/data/object_detection_classes_coco.txt"
net = cv2.dnn.readNetFromTensorflow(path_to_frozen_inference_graph, path_coco_model)
colors = np.random.randint(125, 255, (80, 3))

parent_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(parent_dir)

from enums import FileFormatType, MaterialType, MaterialTitleType, TaskStatus, TaskInfoKey, TrainModelType, TaskType
from task_utils import update_task_info, get_task_info, pre_process, create_task
from log_handler import logger


def get_mask(task_id, node, preprocess_dir, res_dict):
    LABELS = open(class_label).read().strip().split("\n")

    pre_process_files = os.listdir(preprocess_dir)
    pre_process_files_count = len(pre_process_files)

    files_count = len(pre_process_files)
    update_task_info(task_id, TaskInfoKey.PROCESSED_FILE_COUNT.value, files_count)

    for idx, img_name in enumerate(pre_process_files):
        #         print(f"{idx}/{pre_process_files_count}")
        logger.info(f"Task_id:{task_id} Preprocessing FILE: {img_name} current progress: {idx + 1}/{files_count} ")
        update_task_info(task_id, TaskInfoKey.LOG.value,
                         f" Task_id:{task_id} Preprocessing FILE: {img_name} Current Progress {idx}/{files_count}")

        file_name = img_name.split("_page")[0]

        matching_keys = [key for key in res_dict.keys() if file_name in key]
        if not matching_keys:
            continue

        related_pdf_name = matching_keys[0]

        img = cv2.imread(os.path.join(preprocess_dir, img_name))
        height, width, _ = img.shape

        blob = cv2.dnn.blobFromImage(img, swapRB=True)
        net.setInput(blob)
        boxes, masks = net.forward(["detection_out_final", "detection_masks"])
        detection_count = boxes.shape[2]

        result_path = os.path.join("output", "detect", task_id)
        mask_path = os.path.join(result_path, "mask")

        if not os.path.exists(mask_path):
            os.makedirs(mask_path)

        black_image = np.zeros(img.shape, dtype="uint8")

        #         print(height, width, roi_height, roi_width)

        single_res = list()
        for i in range(detection_count):
            box = boxes[0, 0, i]
            class_id = box[1]
            score = box[2]
            if score < 0.5:
                continue

            cur_label = LABELS[int(class_id)]

            x = int(box[3] * width)
            y = int(box[4] * height)
            x2 = int(box[5] * width)
            y2 = int(box[6] * height)

            roi = black_image[y: y2, x: x2]
            roi_height, roi_width, _ = roi.shape

            mask = masks[i, int(class_id)]
            mask = cv2.resize(mask, (roi_width, roi_height))
            _, mask = cv2.threshold(mask, 0.5, 255, cv2.THRESH_BINARY)

            contours, _ = cv2.findContours(np.array(mask, np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            color = colors[int(class_id)]

            cv2.rectangle(img, (x, y), (x2, y2), (int(color[0]), int(color[1]), int(color[2])), 2)

            cv2.putText(img, cur_label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (int(color[0]), int(color[1]), int(color[2])), 2)

            if not res_dict[related_pdf_name].get("result"):
                res_dict[related_pdf_name]["result"] = list()

            detection_res = {
                "class_id": int(class_id),
                "label": cur_label,
                "bbox": [x, y, x2, y2],
                "score": str(score),
            }

            if mask is not None:

                mask_file_path = os.path.join(mask_path, f'{file_name}_mask_{i}.txt')
                np.savetxt(mask_file_path, mask)
                data = np.loadtxt(mask_file_path)


                detection_res["mask_file"] = mask_file_path

            res_dict[related_pdf_name]["result"].append(detection_res)

            update_task_info(task_id, TaskInfoKey.LOG.value,
                             f'Task: {task_id} The file {img_name} predicted current object class is: {cur_label} axis: {x} {y} {x2} {y2}')
            update_task_info(task_id, TaskInfoKey.PROCESSED_FILE_COUNT.value, idx)

            #             print(f'Task: {task_id} The file {img_name} predicted current object class is: {cur_label} axis: {x} {y} {x2} {y2}')
            cv2.imwrite(os.path.join(result_path, 'output.jpg'), img)

        update_task_info(task_id, TaskInfoKey.LOG.value, f'Task: {task_id} The file {img_name} predicted done.')
        update_task_info(task_id, TaskInfoKey.PROCESSED_FILE_COUNT.value, idx + 1)

    return res_dict


def handler(detect_floder, task_id, node):
    logger.info("model loaded")
    update_task_info(task_id, TaskInfoKey.LOG.value, f"model loaded")

    images_path = detect_floder
    preprocess_dir = os.path.join(images_path, "pre_process")
    if not os.path.exists(preprocess_dir):
        os.mkdir(preprocess_dir)

    res_dict = pre_process(task_id, images_path, preprocess_dir)

    res = get_mask(task_id, node, preprocess_dir, res_dict)

    update_task_info(task_id, TaskInfoKey.RESULT.value, res)
    update_task_info(task_id, TaskInfoKey.LOG.value, f"Detection Task: [{task_id}] Already Completed!")
    update_task_info(task_id, TaskInfoKey.STATUS.value, TaskStatus.COMPLETED.value)


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