import os
import cv2
import numpy as np
import pdfplumber
import shutil

path_to_frozen_inference_graph = 'moudules/Mask_RCNN/data/frozen_inference_graph_coco.pb'
path_coco_model = 'moudules/Mask_RCNN/data/mask_rcnn_inception_v2_coco_2018_01_28.pbtxt'
class_label = "moudules/data/object_detection_classes_coco.txt"
net = cv2.dnn.readNetFromTensorflow(path_to_frozen_inference_graph, path_coco_model)
colors = np.random.randint(125, 255, (80, 3))


from enums import FileFormatType, MaterialType, MaterialTitleType, TaskStatus, TaskInfoKey
from task_utils import update_task_info, get_task_info
from log_handler import logger
from extensions import DetectionTaskManager


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
                

def get_mask(task_id, node, pre_process_path, res_dict):
    LABELS = open(class_label).read().strip().split("\n")
    

    pre_process_files = os.listdir(pre_process_path)
    pre_process_files_count = len(pre_process_files)

    files_count = len(pre_process_files)
    update_task_info(task_id, TaskInfoKey.PROCESSED_FILE_COUNT.value, files_count)
    
    for idx, img_name in enumerate(pre_process_files):
#         print(f"{idx}/{pre_process_files_count}")
        logger.info(f"Task_id:{task_id} Preprocessing FILE: {img_name} current progress: {idx+1}/{files_count} ")
        update_task_info(task_id, TaskInfoKey.LOG.value, f" Task_id:{task_id} Preprocessing FILE: {img_name} Current Progress {idx}/{files_count}")
        
        file_name = img_name.split("_page")[0]
   
        matching_keys = [key for key in res_dict.keys() if file_name in key]
        if not matching_keys:
            continue
            
        related_pdf_name = matching_keys[0]
        print(f"模糊匹配到的键：{matching_keys}")


        img = cv2.imread(os.path.join(pre_process_path, img_name))
        height, width, _ = img.shape
        

        blob = cv2.dnn.blobFromImage(img, swapRB=True)
        net.setInput(blob)
        boxes, masks = net.forward(["detection_out_final", "detection_masks"])
        detection_count = boxes.shape[2]

        # Todo: 加日志 idx/img_counts 
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
            print(x, y, x2, y2)
            roi = black_image[y: y2, x: x2]
            roi_height, roi_width, _ = roi.shape  

            
            mask = masks[i, int(class_id)]
            mask = cv2.resize(mask, (roi_width, roi_height))
            _, mask = cv2.threshold(mask, 0.5, 255, cv2.THRESH_BINARY)
            
            
            contours, _ = cv2.findContours(np.array(mask, np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            color = colors[int(class_id)]

            cv2.rectangle(img, (x, y), (x2, y2), (int(color[0]), int(color[1]), int(color[2])), 2)

            # Draw the label on the image
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
                print("mask", type(mask), mask.shape)
                mask_file_path = os.path.join(mask_path ,f'{file_name}_mask_{i}.txt')
                np.savetxt(mask_file_path, mask)
                data = np.loadtxt(mask_file_path)

                print("data", type(data), data.shape)
                detection_res["mask_file"] = mask_file_path    # 保存掩码文件路径
            
            res_dict[related_pdf_name]["result"].append(detection_res)

            update_task_info(task_id, TaskInfoKey.LOG.value, f'Task: {task_id} The file {img_name} predicted current object class is: {cur_label} axis: {x} {y} {x2} {y2}')
            update_task_info(task_id, TaskInfoKey.PROCESSED_FILE_COUNT.value, idx)
            
#             print(f'Task: {task_id} The file {img_name} predicted current object class is: {cur_label} axis: {x} {y} {x2} {y2}')
            cv2.imwrite(os.path.join(result_path, 'output.jpg'), img)
            
        update_task_info(task_id, TaskInfoKey.LOG.value, f'Task: {task_id} The file {img_name} predicted done.')
        update_task_info(task_id, TaskInfoKey.PROCESSED_FILE_COUNT.value, idx +1 )

    return res_dict

def handler(detect_floder, task_id, node):
    logger.info("model loaded")
    update_task_info(task_id, TaskInfoKey.LOG.value, f"model loaded")
    
    images_path = os.path.join("temp_storage", detect_floder)

    pre_process_path = os.path.join(images_path, "pre_process")
    if not os.path.exists(pre_process_path):
        os.mkdir(pre_process_path)
    
    res_dict = pre_process(task_id, images_path, pre_process_path)
    print(res_dict)
    res = get_mask(task_id, node, pre_process_path, res_dict)
    
        
    print(res)
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