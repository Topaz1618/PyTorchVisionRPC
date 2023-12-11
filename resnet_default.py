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
current_directory = os.getcwd()
print("当前工作目录：", current_directory)

from enums import FileFormatType, MaterialType, MaterialTitleType, TaskStatus, TaskInfoKey
from task_utils import update_task_info, get_task_info
from log_handler import logger


def pre_process(task_id, images_path, pre_process_path):
    res_dict = dict()
    file_list = os.listdir(images_path)
    file_count = len(file_list)
    logger.info(f"Start Preprocessing ...")
    update_task_info(task_id, TaskInfoKey.LOG.value, f"Start Preprocessing ...")
    
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
    # 加载图像并进行预处理
    pre_process_files = os.listdir(pre_process_path)
    pre_process_files_count = len(pre_process_files)

    for idx, img_name in enumerate(pre_process_files):
        print(f"Task_id:{task_id} {idx}/{pre_process_files_count} img_name: {img_name}")
        
        file_extension = img_name.split(".")[-1]
        
        if file_extension != "png":
            continue
            
        file_name = img_name.split("_page")[0]
        
        matching_keys = [key for key in res_dict.keys() if file_name in key]
        if not matching_keys:
            continue
            
        related_pdf_name = matching_keys[0]
#         print(f"模糊匹配到的键：{matching_keys}")

#         print("aa", img_name)
        image = Image.open(os.path.join(pre_process_path, img_name))
        image = preprocess(image).unsqueeze(0)  # 增加一个维度，适应模型输入格式
        
        
        # 将模型设置为评估模式
        model.eval()
    
        # 使用模型进行推理
        with torch.no_grad():
            outputs = model(image)
    
        # 获取预测结果
        _, predicted = torch.max(outputs, 1)
        
        
        # 使用softmax获取每个类别的概率分布
        softmax = torch.nn.Softmax(dim=1)
        probabilities = softmax(outputs)
    
        # 获取预测类别的置信度或概率
        confidence = torch.max(probabilities).item()
    
        # 获取所有类别的置信度或概率
        all_probabilities = probabilities.squeeze().tolist()
        
        
        with open("data/imagenet-simple-labels.json") as f:
            class_labels = json.load(f)
        
        predicted_class = predicted.item()
        predicted_label = class_labels[predicted_class]
        
        print("Predicted class index:", predicted_class)
        print("Confidence of predicted class:", confidence)
        print("Predicted class label:", predicted_label)
        
        detection_res = {
            "class_id": int(predicted_class),
            "label": predicted_label,
            "score": confidence,
        }
        
        if not res_dict[related_pdf_name].get("result"):
                res_dict[related_pdf_name]["result"] = list()

        res_dict[related_pdf_name]["result"].append(detection_res)
        
    return res_dict

def handler(detect_floder, task_id, node):
    # 加载预训练的 ResNet50 模型
    # 创建 ResNet50 模型实例（不包括预训练权重）
    model = models.resnet50(pretrained=False)
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # 加载本地保存的模型参数
    # model.load_state_dict(torch.load('Resnet/pretrain_models/resnet50.pth', map_location=torch.device('cpu')))
    model.load_state_dict(torch.load('moudules/Resnet/pretrain_models/resnet50.pth', map_location=device))
    model.eval()  # 设置模型为评估模式
    print("Model loaded")
    
    # 图像预处理函数
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    images_path = os.path.join("temp_storage", detect_floder)
    
    pre_process_path = os.path.join(images_path, "pre_process")
    if not os.path.exists(pre_process_path):
        os.mkdir(pre_process_path)

    # 预测图像类别

    res_dict = pre_process(images_path, pre_process_path)
    res = predict_image_class(task_id, images_path, pre_process_path, model, preprocess, res_dict)
    print(res)
    
    # Todo: 释放资源， 结果增加到 MongoDB


if __name__ == "__main__":
    detect_floder = "detect_demo1"
    task_id = "123"
    node = "worker1"
    handler(detect_floder, task_id, node)
    
