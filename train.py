import os
import uuid
import base64
import asyncio
import os
import argparse
from time import sleep, time
from datetime import datetime

from Crypto.Cipher import AES
from Crypto.Random import get_random_bytes
from Crypto.Util.Padding import pad
from pymongo import MongoClient
from gridfs import GridFS
import motor.motor_asyncio

from log_handler import logger
from config import MONGODB_SERVERS, replicaSet, TEMP_STORAGE_DIR
from task_utils import get_task_info, update_task_info, remove_used_capacity, create_task
from compression_utils import decompress_zip
from enums import DetectionAlgorithm, TaskType, TaskInfoKey, ModelSource, TaskType, DetectionAlgorithm, TrainModelType
from extensions import DetectionTaskManager, TrainingTaskManager

from moudules.YOLO.ultralytics_yolov5_master.train_wrapper import yolo_train_wrapper as yolo_handler
from moudules.paddle.train_wrapper import paddle_train_wrapper as paddle_wrapper
from moudules.Mask_RCNN.train_wrapper import maskrcnn_wrapper as maskrcnn_wrapper
from moudules.Resnet.train import start as resnet_wrapper


secret_key = b'YourSecretKey123'
parser = argparse.ArgumentParser(description='Description of your script')
parser.add_argument('--task_id', type=str, help='ID of the detect task')
parser.add_argument('--model', type=str, default='YOLO', choices=['YOLO', 'Mask_RCNN', 'PaddleOCR', 'Resnet'],
                    help='Detection algorithm (default: Tesseract)')
parser.add_argument('--node', type=str, help='Name of the node')
parser.add_argument('--dataset', type=str, help='Name of the dataset')
parser.add_argument('--epoch', type=int, default=100, help='Number of epochs (default: 100)')
parser.add_argument('--batch_size', type=int, default=32, help='Batch size (default: 32)')
parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate value (default: 0.001)')
args = parser.parse_known_args()[0]

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
                update_task_info(task_id, TaskInfoKey.LOG.value,
                                 f"Task: [{task_id}] Download {filename} chunk: {chunk_number}/{len(versions)}")
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


async def async_download(task_id, grid_save_name, detect_temp_path):
    if not os.path.exists(detect_temp_path):
        gridfs_manager = AsyncGridFSManager()
        await gridfs_manager.download_file(task_id, grid_save_name, detect_temp_path)


def main():
    # 创建参数解析器

    task_id = str(uuid.uuid4())  # 生成唯一的任务ID
    create_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    epoch = 2
    username = "18310703270"
    model = TrainModelType.RESNET.value
    batch_size = 32
    learning_rate = 0.01
    node = "worker1"
    dataset = "coco"

    task = create_task(task_id, TaskType.TRAINING.value, create_time, model, dataset, epoch, batch_size, learning_rate)

    task_obj = TrainingTaskManager()
    task_obj.create_task(task_id, username, model, dataset, epoch, batch_size, learning_rate, create_time)
    task_obj.close()
    """

    task_id = args.task_id
    epoch = args.epoch
    batch_size = args.batch_size
    learning_rate = args.learning_rate
    node = args.node
    dataset = args.dataset

    if dataset == "coco_mini":
        dataset = "mini"

    model = args.model
    """
    # from ipdb import set_trace; set_trace()
    if model == TrainModelType.RESNET.value:
        #         resnet_wrapper(task_id, epoch, batch_size, learning_rate, dataset, node)
        res = resnet_wrapper.start(task_id, epoch, batch_size, learning_rate, dataset, node)

    elif model == TrainModelType.YOLO.value:
        res = yolo_handler(task_id, epoch, batch_size, learning_rate)

    elif model == TrainModelType.MaskRCNN.value:
        res = maskrcnn_wrapper.maskrcnn_wrapper(task_id, epoch, batch_size, learning_rate)

    elif model == TrainModelType.PADDLEOCR.value:
        res = paddle_wrapper.paddle_train_wrapper(task_id, epoch, batch_size, learning_rate, dataset, node)

    else:
        raise ValueError("invaild paramaters")

    if res:
        logger.warning(f"Task {task_id} Result: {res}")

        remove_used_capacity(node, TaskType.TRAINING.value)


if __name__ == "__main__":
    main()
