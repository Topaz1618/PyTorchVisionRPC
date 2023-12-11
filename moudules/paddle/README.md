# Paddle 

## 推荐环境
PaddlePaddle >= 2.1.2
Python 3.7
CUDA10.1 / CUDA10.2
CUDNN 7.6

## 安装
您的机器安装的是CUDA9或CUDA10，请运行以下命令安装
```
python3 -m pip install paddlepaddle-gpu -i https://mirror.baidu.com/pypi/simple
```

您的机器是CPU，请运行以下命令安装
```
python3 -m pip install paddlepaddle -i https://mirror.baidu.com/pypi/simple
```

安装 Paddleocr
```
pip install "paddleocr>=2.0.1" # 推荐使用2.0.1+版本
```

# 使用

## 训练
python3 train_wrapper.py


## 自定义推理模型
python3 detect_wrapper.py


# 使用(弃用)

## 训练
python3 train.py -c configs/ch_PP-OCRv3_rec_distillation.yml -o Global.pretrained_model=./pretrain_models/ch_PP-OCRv3_rec_train/best_accuracy.pdparams


python3 train.py -c configs/en_PP-OCRv3_rec.yml -o Global.pretrained_model=./pretrain_models/en_PP-OCRv3_rec_train/best_accuracy

## 测试当前指定预训练模型
python3 infer_rec.py -c configs/en_PP-OCRv3_rec.yml -o Global.pretrained_model=pretrain_models/ch_PP-OCRv3_rec_slim_train/best_accuracy Global.infer_img=data/t.png


## 自定义推理模型
python3 predict_rec.py --image_dir="data/cn_word.png" --rec_model_dir="models/ch_PP-OCRv3_rec_infer"


