import torch
import sys

import random
import torch
import cv2

import numpy as np
from pathlib import Path

# from ipdb import set_trace; set_trace();
# sys.path.append('ultralytics_yolov5_master')  
# from models import Model 



def pre_process(images_path, pre_process_path):
    res_dict = dict()
    file_list = os.listdir(images_path)

    
    for idx, filename in enumerate(file_list):
        file_extension = filename.split(".")[-1]
        file_name = filename.split(".")[0]
        res_dict[filename] = {"file_type": file_extension.upper()}

        detection_type_list = ["pdf", "png", "jpg", "jpeg"]
        img_extension_list = ["png", "jpg", "jpeg"]
        if file_extension not in detection_type_list:
            continue
            
        if file_extension in img_extension_list:
            shutil.copy(os.path.join(images_path, filename), os.path.join(pre_process_path, f"{file_name}_page_0.png"))
        else:
            
            with pdfplumber.open(os.path.join(images_path, filename)) as pdf:
                for page_num, page in enumerate(pdf.pages):
                    img = page.to_image()
                    pdf_image = page.to_image().original
    
                    # 将原始图像数据转换为 OpenCV 图像对象
                    open_cv_image = cv2.cvtColor(np.array(pdf_image), cv2.COLOR_RGB2BGR)
                    cv2.imwrite(os.path.join(pre_process_path, f"{file_name}_page_{page_num}.png"), open_cv_image)
                
    return res_dict


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


# 设置缓存目录
cache_dir = '/mnt/Demo/YOLO/'
torch.hub.set_dir(cache_dir)

colors = np.random.randint(125, 255, (80, 3))

# 定义模型权重文件路径
model_weights_path = '../yolov5s.pt'

# 加载模型
# model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_weights_path)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
 
# Model
# ../: 源目录, 查找 hubconf.py, 模型文件等，'yolov5s'：找模型文件，没有会下载  ; source='local' 本地加载模型
model = torch.hub.load('../', 'yolov5s', source='local')  # or yolov5m, yolov5l, yolov5x, custom
model = model.to(device)



# 加载图像（或视频帧）进行目标检测
image_path = 'bus.jpg'
img = cv2.imread(image_path)

# 进行目标检测
results = model(img)

# 显示检测结果
# results.show()
color = colors[int(random.randint(1, 10))]
print("color", color)

# 或者获取检测到的对象信息并进行后续处理
detected_objects = results.pandas().xyxy[0]
print(detected_objects)

for index, obj in detected_objects.iterrows():
    x1, y1, x2, y2, conf, label = int(obj[0]), int(obj[1]), int(obj[2]), int(obj[3]), obj[4], int(obj[5])
    print(f"x1:{x1} x2:{x2} y1:{y1} y2:{y2} conf:{conf} Label: {label}")
#     画出边界框
    cv2.rectangle(img, (x1, y1), (x2, y2), (int(color[0]), int(color[1]), int(color[2])), 2)

#     # 标签文本
    label_text = f"{model.names[label]}: {conf:.2f}"

#     # 在边界框上方显示类别标签和置信度
    cv2.putText(img, label_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (int(color[0]), int(color[1]), int(color[2])), 2)
    
cv2.imwrite('output.jpg', img)
