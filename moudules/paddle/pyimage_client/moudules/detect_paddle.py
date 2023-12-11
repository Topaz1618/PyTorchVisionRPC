import paddle
from paddleocr import PaddleOCR, draw_ocr

# 加载模型
ocr = PaddleOCR(use_gpu=False)

# 输入图片
img_path = '你的户口本图片路径'
result = ocr.ocr(img_path, use_gpu=False)

# 打印结果
for line in result:
    line_text = ' '.join([word_info[-1] for word_info in line])
    print(line_text)