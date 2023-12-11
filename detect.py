"""
For test
"""
import os
import uuid
import base64
import argparse
import asyncio
from datetime import datetime
from Crypto.Cipher import AES
from Crypto.Random import get_random_bytes
from Crypto.Util.Padding import pad

from pymongo import MongoClient
import asyncio
from gridfs import GridFS
import motor.motor_asyncio

from time import sleep, time

from task_utils import get_task_info, update_task_info, remove_used_capacity, create_task
from compression_utils import decompress_zip
# from enum import Enum
from enums import DetectionAlgorithm, TaskType, TaskInfoKey, ModelSource
from moudules.default_detect import pydetect_loop
from moudules.detect_paddle import detect_img
from moudules.yolo_default import handler as yolo_handler
from moudules.maskrcnn_default import handler as maskrcnn_handler
from moudules.resnet_default import handler as resnet_handler
from moudules.paddle_default import handler as paddle_handler


from log_handler import logger
from config import MONGODB_SERVERS, replicaSet, TEMP_STORAGE_DIR
from extensions import DetectionTaskManager




secret_key = b'YourSecretKey123'

# 创建参数解析器
parser = argparse.ArgumentParser(description='Description of your script')

# 添加需要的参数
parser.add_argument('--task_id', type=str, help='Id of detect task')
parser.add_argument('--model', type=str, default='PaddleOCR', choices=['YOLO', 'Mask_RCNN', 'PaddleOCR', 'Resnet'], help='Detection algorithm')
parser.add_argument('--node', type=str, help='Id of detect task')


# 解析命令行参数
args = parser.parse_args()


secret_key = b'YourSecretKey123'

def async_conn_db():
    """
    Establishes an asynchronous connection to the MongoDB server and returns the async client and database objects
    :return:
        async_client: The asynchronous connection to the MongoDB server
        async_db: The database object for the specified file system
    """
    mongo_cluster_uri = f"mongodb://{','.join(MONGODB_SERVERS)}/?replicaSet={replicaSet}"
    async_client = motor.motor_asyncio.AsyncIOMotorClient(mongo_cluster_uri)
    async_db = async_client['filesystem']
    return async_client, async_db


def decrypt_data(encrypted_text):
    encrypted_text = base64.b64decode(encrypted_text)

    cipher = AES.new(secret_key, AES.MODE_CBC, encrypted_text[:AES.block_size])
    decrypted = cipher.decrypt(encrypted_text[AES.block_size:])
    # Strip PKCS7 padding manually

    padding_length = decrypted[-1]
    decrypted = decrypted[:-padding_length]
    return decrypted

def encrypt_data(data):
    cipher = AES.new(secret_key, AES.MODE_CBC)
    iv = cipher.iv  # Get the IV generated by PyCryptodome
    if not isinstance(data, bytes):
        data = data.encode('utf-8')

    padded_data = pad(data, AES.block_size)
    ct_bytes = cipher.encrypt(padded_data)

    # Combine IV and ciphertext and encode as Base64
    iv_and_ciphertext = iv + ct_bytes
    return base64.b64encode(iv_and_ciphertext).decode('utf-8')


class MongoDBManager:
    def __init__(self):
        self.async_client, self.async_db = async_conn_db()
        self.fs = motor.motor_asyncio.AsyncIOMotorGridFSBucket(self.async_db)


class AsyncGridFSManager(MongoDBManager):
    """
    A manager class for handling asynchronous operations with GridFS in MongoDB.
    """

    def __init__(self):
        super().__init__()
        self.chunk_size = 1024 * 1024

    async def download_chunk(self, chunk_id):
        download_stream = await self.fs.open_download_stream(chunk_id)
        return download_stream

    async def upload_chunk(self, data, filename):
        upload_stream = self.fs.open_upload_stream(filename=filename)
        try:
            await upload_stream.write(data)

        finally:
            await upload_stream.close()
        print(f'Uploaded file {filename} to GridFS')

    async def count_file_clips(self, filename):
        clip_count = await self.fs.find({'filename': filename}).to_list(length=None)
        return clip_count

    async def upload_file(self, filename):
        with open(filename, 'rb') as file:

            while True:
                chunk_data = file.read(self.chunk_size)
                if not chunk_data:
                    break
                print(f"Chunk data: {chunk_data}")
                encrypted_data = encrypt_data(chunk_data)
                if isinstance(encrypted_data, str):
                    # Encode the string data to bytes (using UTF-8 encoding in this example)
                    encrypted_data = encrypted_data.encode('utf-8')
                print(f"Encrypted data: {encrypted_data}")
                await self.upload_chunk(encrypted_data, filename)

            print(f'Uploaded file {filename} to GridFS')

    async def download_file(self, task_id, filename, output_path):
        versions = await self.count_file_clips(filename)

        print(versions)
        with open(output_path, 'wb') as file:
            for chunk_number in range(len(versions)):
                version = versions[chunk_number]  # 指定切片
                chunk_id = version['_id']
                update_task_info(task_id, TaskInfoKey.LOG.value, f"Task: [{task_id}] Download {filename} chunk: {chunk_number}/{len(versions)}")
                logger.info(f"Task: [{task_id}] Download {filename} chunk: {chunk_number}/{len(versions)}")
                download_stream = await self.download_chunk(chunk_id)
                data = await download_stream.read()

                # print(f"Encrypted data: {data}")
                if not data:
                    break

                chunk_data = decrypt_data(data)
#                 print(f"Chunk data: {len(chunk_data)}")

                file.write(chunk_data)

        update_task_info(task_id, TaskInfoKey.LOG.value, f"Starting Unzipping: {filename}")
        uncompress_file_path = decompress_zip(task_id, output_path)

        logger.info(f'Downloaded file {filename} from GridFS to {output_path}')
        update_task_info(task_id, TaskInfoKey.LOG.value, f"Downloaded file {filename} from GridFS to {output_path}")

    async def delete_files_async(self, filename):
        versions = await self.fs.find({'filename': filename}).to_list(length=None)

        for version in versions:
            chunk_id = version['_id']
            await self.fs.delete(chunk_id)
            
            

async def async_download(task_id,grid_save_name, detect_temp_path):
    if not os.path.exists(detect_temp_path):
        gridfs_manager = AsyncGridFSManager()
        await gridfs_manager.download_file(task_id, grid_save_name, detect_temp_path)
           
            
def main():
    args = parser.parse_args()
    # For test 
    task_id = args.task_id
#     task_id = str(uuid.uuid4())  # 生成唯一的任务ID
    create_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    model = args.model
    node = args.node

    # task = create_task(task_id, TaskType.DETECT.value, create_time, model, 'detect_demo1.zip')
    # task_obj = DetectionTaskManager()
    # task_obj.create_task(task_id, "18310703270", model , 'detect_demo1.zip', create_time)
    # task_obj.close()


    if not args.task_id:
        raise ValueError("no task id")

    if not args.node:
        raise ValueError("no node ")

    task_id = args.task_id
    node = args.node
    model = args.model

    task_info = get_task_info(task_id)
    zip_name = task_info.get("detect_file")
    detect_temp_path = os.path.join(TEMP_STORAGE_DIR, zip_name)

    download_result = asyncio.run(async_download(task_id, f'{task_id}_{zip_name}', detect_temp_path))
    update_task_info(task_id, TaskInfoKey.LOG.value, f"Starting Unzipping: {zip_name}")
    detect_floder = decompress_zip(task_id, detect_temp_path)

#     res = handler(detect_floder, task_id, node)
    
    logger.info(f"Starting Unzipping: {zip_name}")
    update_task_info(task_id, TaskInfoKey.LOG.value, f"Starting Unzipping: {zip_name}")
    detect_floder = decompress_zip(task_id, detect_temp_path)

    print(model, detect_floder)

#     if model == DetectionAlgorithm.TESSERACT.value:
#         res = pydetect_loop(task_id, uncompress_file_path)

    if model == DetectionAlgorithm.YOLO.value:
        res = yolo_handler(detect_floder, task_id, node)
        
    elif model == DetectionAlgorithm.MaskRCNN.value:
        res = maskrcnn_handler(detect_floder, task_id, node)

    elif model == DetectionAlgorithm.RESNET.value:
        res = resnet_handler(detect_floder, task_id, node)

    elif model == DetectionAlgorithm.PADDLEOCR.value:
        res = paddle_handler(detect_floder, task_id, node)

    else:
        res = None

    if res:
        logger.warning(f"Task {task_id} Result: {res}")
        remove_used_capacity(node, TaskType.DETECT.value)


if __name__ == "__main__":
    # 指定PDF文件路径
    print(" === 文件识别任务调试用方法测试开始 ===")
    main()
    # asyncio.run()

    print(" === 文件识别任务调试用方法测试结束 ===")