# Resnet 训练

## 基于默认预训练模型训练
python train.py


# Resnet 识别
## 使用默认模型识别
python detect.py

## 使用指定模型识别
python detect.py 
    image_path = 'dataset/val/bees/72100438_73de9f17af.jpg'
    model_path = "pretrain_models/resnet_epoch_3.pt"
    
    res = resnet_detect(model_path, image_path)
    print(f'The predicted class is: {class_names[res.item()]}')