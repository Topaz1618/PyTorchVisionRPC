import os
import uuid
import base64
import asyncio
from datetime import datetime
from Crypto.Cipher import AES
from Crypto.Random import get_random_bytes
from Crypto.Util.Padding import pad


from pymongo import MongoClient
from gridfs import GridFS
import motor.motor_asyncio

# from moudules.maskrcnn_default import handler
from moudules.resnet_default import handler
# from moudules.yolo_default import handler
# from moudules.paddle_default import handler

from task_utils import create_task
from enums import TaskType, DetectionAlgorithm
from extensions import DetectionTaskManager, MongoDBManager

from log_handler import logger
from config import MONGODB_SERVERS, replicaSet, TEMP_STORAGE_DIR
from task_utils import get_task_info, update_task_info, remove_used_capacity, create_task
from compression_utils import decompress_zip
from enums import DetectionAlgorithm, TaskType, TaskInfoKey, ModelSource
from compression_utils import decompress_zip



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
    zip_name = 'detect_demo1.zip'
    task_id = str(uuid.uuid4())  # 生成唯一的任务ID
    create_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    task = create_task(task_id, TaskType.DETECT.value, create_time, DetectionAlgorithm.YOLO.value , 'detect_demo1.zip')
    
    task_obj = DetectionTaskManager()
    task_obj.create_task(task_id, "18310703270", DetectionAlgorithm.YOLO.value, 'detect_demo1.zip', create_time)
#     task_obj.close()
        
#     task_id = "3e48b924-3fae-4225-ab3f-8dd7721b82a8"
        
    detect_temp_path = os.path.join(TEMP_STORAGE_DIR, zip_name)
    
    download_result = asyncio.run(async_download(task_id, f'{task_id}_{zip_name}', detect_temp_path))
    update_task_info(task_id, TaskInfoKey.LOG.value, f"Starting Unzipping: {zip_name}")
    detect_floder = decompress_zip(task_id, detect_temp_path)


    
    print(detect_floder)
    
#     detect_floder = "detect_demo1"
    # task_id = "123"
    node = "worker1"
    res = handler(detect_floder, task_id, node)
    
#     task_obj = DetectionTaskManager()
#     task_obj.update_task(task_id, TaskInfoKey.RESULT.value, res)
#     task_obj.update_task(task_id, "task_status", TaskStatus.COMPLETED.value)
#     task_obj.close() 

    
if __name__ == "__main__":
    main()