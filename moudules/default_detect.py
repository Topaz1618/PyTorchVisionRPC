import os
import sys
import shutil
import uuid

import pdfplumber
import pytesseract
import re
import docx
from enum import Enum
from time import sleep, time


sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath("."))))

from enums import FileFormatType, MaterialType, MaterialTitleType, TaskStatus, TaskInfoKey
from config import TEMP_STORAGE_DIR
from task_utils import update_task_info, get_task_info
from log_handler import logger


def match_file_format(file_extension):
    matching_format = None
    for format_type in FileFormatType:
        if format_type.value == file_extension:
            matching_format = format_type
            text_format = matching_format.get_text_format()
            return text_format

    if not matching_format:
        return FileFormatType.UNKNOWN.get_text_format()


def extract_table_data(task_id, pdf_path):
    # 使用pdfplumber打开PDF文件
    print("pdf_path", pdf_path)
    with pdfplumber.open(pdf_path) as pdf:
        page_num = len(pdf.pages)

        table_data = []
        current_page = 1
        for page in pdf.pages:
            logger.info(f"Processing Task: [{task_id}], File: {pdf_path}, Page: {current_page}/{page_num}")
            update_task_info(task_id, TaskInfoKey.LOG.value, f"Processing Task: [{task_id}], File: {pdf_path}, Page: {current_page}/{page_num}")
            # 提取页面中的表格
            tables = page.extract_tables()
            for table in tables:
                # 将表格数据转换为JSON格式
                table_json = []
                for row in table:
                    table_json.append(row)
                table_data.append(table_json)

            logger.info(f"Processing Task: [{task_id}], File: {pdf_path}, Completed Page: {current_page}/{page_num}")
            update_task_info(task_id, TaskInfoKey.LOG.value, f"Processing Task: [{task_id}], File: {pdf_path}, Completed Page: {current_page}/{page_num}")

            current_page += 1
            sleep(1)

    return table_data, page_num


def extract_text(task_id, pdf_path):
    # 使用pdfplumber打开PDF文件
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            # 将pdfplumber的Page对象转换为图像
            img = page.to_image()

            # 使用OCR来提取图像中的文本
            ocr_text = pytesseract.image_to_string(img.original, lang='chi_sim')

            # 使用正则表达式匹配身份证号码
            id_card_pattern = re.compile(r'\d{17}[\dXx]|\d{15}|\d{18}')
            matches = re.findall(id_card_pattern, ocr_text)

            # 打印匹配到的身份证号码
            if matches:
                for match in matches:
                    logger.info(f"身份证号码: {match}")
                    return match


def pydetect_loop(task_id, detect_folder_path):
    file_dic = {}
    file_list = os.listdir(detect_folder_path)
    total_file_count = len(file_list)
    update_task_info(task_id, TaskInfoKey.TOTAL_FILE_COUNT.value, total_file_count)
    processed_file_count = 0

    idx = 0
    for file_name in file_list:

        update_task_info(task_id, TaskInfoKey.LOG.value, f"{task_id} Processing file: {file_name}")
        file_path = os.path.join(detect_folder_path, file_name)

        file_extension = os.path.splitext(file_path)[1]
        file_type = match_file_format(file_extension)
        file_dic[file_name] = {
            "file_type": file_type,
        }

        if file_extension == FileFormatType.PDF.value:
            table_data, page_num = extract_table_data(task_id, file_path)

            file_dic[file_name]["page_num"] = page_num

            if not table_data:
                id_number = extract_text(task_id, file_path)
                if id_number:
                    file_dic[file_name]["material_type"] = MaterialType.ID_CARD.value
                    table_data = {"id": id_number}

            else:
                table_data = [[item.replace('\n', '') for item in sublist if item is not None and item != ''] for sublist in table_data[0]]

            file_dic[file_name]["json_data"] = table_data

            if not table_data:
                continue

            if any(MaterialTitleType.LAND_MAP.value in item for item in table_data):
                file_dic[file_name]["material_type"] = MaterialType.LAND_MAP.value

            elif any(MaterialTitleType.REAL_ESTATE_APPLICATION.value in item for item in table_data):
                file_dic[file_name]["material_type"] = MaterialType.REAL_ESTATE_APPLICATION.value

            else:
                file_dic[file_name]["material_type"] = MaterialType.UNKNOWN.value

        processed_file_count += 1

        update_task_info(task_id, TaskInfoKey.LOG.value, f"Processing Task: [{task_id}]  File detection completed : {file_name}")
        update_task_info(task_id, TaskInfoKey.PROCESSED_FILE_COUNT.value, processed_file_count)

        sleep(1)

        idx += 1
    update_task_info(task_id, TaskInfoKey.RESULT.value, file_dic)
    # Todo: upload to MongoDB
    update_task_info(task_id, TaskInfoKey.LOG.value, f"Detection Task: [{task_id}] Already Completed!")
    update_task_info(task_id, TaskInfoKey.STATUS.value, TaskStatus.COMPLETED.value)
    return True


if __name__ == "__main__":
    task_id = "4ba5ff7d-f43c-4676-9dd8-57dc4339f1d6"
    detect_folder_path = os.path.join(TEMP_STORAGE_DIR, "detect_demo1")
    pydetect_loop(task_id, detect_folder_path)