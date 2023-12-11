import os
import sys
import json
import shutil

import random
import torch
import cv2
import pdfplumber

import numpy as np
from pathlib import Path

colors = np.random.randint(125, 255, (80, 3))

import torch
import torchvision.models as models
from torchvision import transforms
from PIL import Image

parent_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(parent_dir)

from enums import FileFormatType, MaterialType, MaterialTitleType, TaskStatus, TaskInfoKey
from task_utils import update_task_info, get_task_info
from log_handler import logger
from extensions import DetectionTaskManager


# 图片路径
# image_path = 'bus.jpg'  # 替换为你自己的图片路径


def pre_process(task_id, images_path, pre_process_path):
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
            shutil.copy(os.path.join(images_path, filename), os.path.join(pre_process_path, f"{file_name}_page_0.png"))
        else:

            with pdfplumber.open(os.path.join(images_path, filename)) as pdf:
                for page_num, page in enumerate(pdf.pages):
                    img = page.to_image()
                    pdf_image = page.to_image().original

                    open_cv_image = cv2.cvtColor(np.array(pdf_image), cv2.COLOR_RGB2BGR)
                    cv2.imwrite(os.path.join(pre_process_path, f"{file_name}_page_{page_num}.png"), open_cv_image)

    return res_dict


# 图像分类函数
def predict_image_class(task_id, images_path, pre_process_path, model, preprocess, res_dict):
    pre_process_files = os.listdir(pre_process_path)
    pre_process_files_count = len(pre_process_files)

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

        image = Image.open(os.path.join(pre_process_path, img_name))
        image = preprocess(image).unsqueeze(0)
        model.eval()

        with torch.no_grad():
            outputs = model(image)

        _, predicted = torch.max(outputs, 1)

        softmax = torch.nn.Softmax(dim=1)
        probabilities = softmax(outputs)
        confidence = torch.max(probabilities).item()
        all_probabilities = probabilities.squeeze().tolist()

        with open("moudules/data/imagenet-simple-labels.json") as f:
            class_labels = json.load(f)

        predicted_class = predicted.item()
        predicted_label = class_labels[predicted_class]

        #         print("Predicted class index:", predicted_class)
        #         print("Confidence of predicted class:", confidence)
        #         print("Predicted class label:", predicted_label)

        detection_res = {
            "class_id": int(predicted_class),
            "label": predicted_label,
            "score": confidence,
        }

        if not res_dict[related_pdf_name].get("result"):
            res_dict[related_pdf_name]["result"] = list()

        res_dict[related_pdf_name]["result"].append(detection_res)
        update_task_info(task_id, TaskInfoKey.LOG.value,
                         f'Task: {task_id} The file {img_name} predicted current object class is: {predicted_label} confidence: {confidence}')
        update_task_info(task_id, TaskInfoKey.LOG.value, f'Task: {task_id} The file {img_name} predicted done.')
        update_task_info(task_id, TaskInfoKey.PROCESSED_FILE_COUNT.value, idx + 1)
        logger.info(f'Task: {task_id} The file {img_name} predicted done.')

    return res_dict


def handler(detect_floder, task_id, node):
    model = models.resnet50(pretrained=False)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # model.load_state_dict(torch.load('Resnet/pretrain_models/resnet50.pth', map_location=torch.device('cpu')))
    model.load_state_dict(torch.load('moudules/Resnet/pretrain_models/resnet50.pth', map_location=device))
    model.eval()
    logger.info("model loaded")
    update_task_info(task_id, TaskInfoKey.LOG.value, f"model loaded")

    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    images_path = detect_floder
    #     images_path = os.path.join("temp_storage", detect_floder)
    pre_process_path = os.path.join(images_path, "pre_process")
    if not os.path.exists(pre_process_path):
        os.mkdir(pre_process_path)

    res_dict = pre_process(task_id, images_path, pre_process_path)
    res = predict_image_class(task_id, images_path, pre_process_path, model, preprocess, res_dict)

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

