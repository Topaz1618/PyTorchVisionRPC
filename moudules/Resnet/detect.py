import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
from PIL import Image
from tempfile import TemporaryDirectory


data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

image_datasets = {x: datasets.ImageFolder(os.path.join('dataset', x),
                                          data_transforms[x])
                  for x in ['train', 'val']}

class_names = image_datasets['train'].classes
print(class_names)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def resnet_detect(images_path, task_id, node, model_name=None):
    if not model_name:
        print(f"Model name is not provided, use default model")
        model_path = "models/resnet_epoch_3.pt"
    else:
        model_path = os.path.join("models", model_name)
        if not os.path.exists(model_path):
            print(f"{model_name} does not exist, use default model")
            model_path = "models/resnet_epoch_3.pt"

    # 加载模型
    model = models.resnet50(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, len(class_names))
    model.load_state_dict(torch.load(model_path))
    model = model.to(device)
    model.eval()

    # Todo: 加 log
    print(f'Model loaded')

    # 加载图像
    res = []

    all_images = os.listdir(images_path)
    img_count = len(all_images)
    for idx, img_name in enumerate(all_images):
        # Todo: 加redis idx/img_count
        

        image_path = os.path.join(images_path, img_name)
        image = Image.open(image_path)
        image = data_transforms['val'](image).unsqueeze(0).to(device)

        # 预测
        outputs = model(image)
        _, preds = torch.max(outputs, 1)

        if preds:
            label = class_names[preds.item()]
        else:
            label = "unknown"

        res.append({"img_name": img_name, "class": {label}})

        # Todo: 加日志 log  f"{img}" 处理已完成，预测类别为 {label}
        print(f'The file {img_name} predicted class is: {label}')

    print("res", res)

    # Todo: 结果保存到 MongoDB


if __name__ == "__main__":
    images_path = 'dataset/val/bees'
    node = "worker1"
    task_id = "123"
    res = resnet_detect(images_path, task_id, node)
