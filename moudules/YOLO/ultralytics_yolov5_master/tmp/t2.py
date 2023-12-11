import torch
import cv2
import numpy as np
from pathlib import Path


colors = np.random.randint(125, 255, (80, 3))


# 定义模型权重文件路径
model_weights_path = './yolov5s.pt'

# 加载模型
model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_weights_path)

# 加载图像（或视频帧）进行目标检测
image_path = 'bus.jpg'
img = cv2.imread(image_path)

# 进行目标检测
results = model(img)

# 显示检测结果
# results.show()
color = colors[int(1)]

# 或者获取检测到的对象信息并进行后续处理
detected_objects = results.pandas().xyxy[0]
print(detected_objects)

for obj in detected_objects:
    # 获取边界框坐标和类别信息
    x1, y1, x2, y2, conf, label = int(obj[0]), int(obj[1]), int(obj[2]), int(obj[3]), obj[4], int(obj[5])

    # 画出边界框
    cv2.rectangle(img, (x1, y1), (x2, y2), (int(color[0]), int(color[1]), int(color[2])), 2)

    # 标签文本
    label_text = f"{model.names[label]}: {conf:.2f}"

    # 在边界框上方显示类别标签和置信度
    cv2.putText(img, label_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (int(color[0]), int(color[1]), int(color[2])), 2)
    
    cv2.imwrite('output.jpg', img)
