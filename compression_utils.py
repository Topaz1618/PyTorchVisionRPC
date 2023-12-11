import os
import tarfile
import zipfile
import shutil

from task_utils import update_task_info
from config import TEMP_STORAGE_DIR
from enums import TaskInfoKey


def compress_to_tar(source_folder, tar_file_path):
    """
     压缩指定文件夹为 tar 文件

     Args:
         source_folder (str): 要压缩的文件夹路径
         tar_file_path (str): 目标 tar 文件路径

     """
    with tarfile.open(tar_file_path, "w") as tar:
        tar.add(source_folder, arcname="")


def decompress_tar(tar_file_path, extract_to_path):
    """
    解压缩 tar 文件到指定目录

    Args:
        tar_file_path (str): 要解压缩的 tar 文件路径
        extract_to_path (str): 解压缩目标路径

    """
    with tarfile.open(tar_file_path, "r") as tar:
        tar.extractall(extract_to_path)


def compress_to_zip(source_folder, zip_file_path):
    """
    压缩指定文件夹为 zip 文件

    Args:
        source_folder (str): 要压缩的文件夹路径
        zip_file_path (str): 目标 zip 文件路径

    """
    with zipfile.ZipFile(zip_file_path, "w", zipfile.ZIP_DEFLATED) as zip:
        for root, dirs, files in os.walk(source_folder):
            for file in files:
                file_path = os.path.join(root, file)
                zip.write(file_path, arcname=os.path.relpath(file_path, source_folder))


def decompress_zip(task_id, zip_file_path):
    """
    解压缩 zip 文件到指定目录

    Args:
        zip_file_path (str): 要解压缩的 zip 文件路径
        extract_to_path (str): 解压缩目标路径

    """

    folder_name = zip_file_path.split("/")[-1].split(".zip")[0]
    print("!!folder_name", folder_name)
    res_path = os.path.join(TEMP_STORAGE_DIR, folder_name)

    if not os.path.exists(res_path):
        os.mkdir(res_path)

    with zipfile.ZipFile(zip_file_path, 'r') as zf:

        for fn in zf.namelist():
            right_fn = fn.encode('cp437').decode('utf-8')  # 将文件名正确编码

            if f"{folder_name}/" == right_fn or right_fn.startswith("__"):
                continue

            print(f"Task ID: {task_id} Uncompressed file: {right_fn}")
            with open(os.path.join(TEMP_STORAGE_DIR, right_fn), 'wb') as output_file:  # 创建并打开新文件
                with zf.open(fn, 'r') as origin_file:  # 打开原文件
                    shutil.copyfileobj(origin_file, output_file)  # 将原文件内容复制到新文件
            print(f"解压文件  {right_fn} 完成")
            update_task_info(task_id, TaskInfoKey.LOG.value, f"解压文件  {right_fn} 完成")


    return res_path
