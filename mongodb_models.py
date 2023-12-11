import motor.motor_asyncio
from pymongo import MongoClient, ASCENDING

from config import MONGODB_SERVERS, replicaSet


import motor.motor_asyncio
from pymongo import MongoClient, ASCENDING

from config import MONGODB_SERVERS, replicaSet, FileSystem, GRIDFS_COLLECTION_NAME
from enums import TaskStatus, UploadStatus, ModelSource


def _mongo_server_uri():
    """
    Generates the MongoDB server URI based on the configuration settings
    """
    mongodb_servers = ",".join(MONGODB_SERVERS)
    mongo_cluster_uri = f"mongodb://{mongodb_servers}/?replicaSet={replicaSet}"
    return mongo_cluster_uri


def conn_db():
    """
    Establishes a connection to the MongoDB server and returns the client and database objects
    :return:
        client: The connection to the MongoDB server
        db: The database object for the specified file system
    """
    mongo_cluster_uri = _mongo_server_uri()
    client = MongoClient(mongo_cluster_uri)
    db = client[FileSystem]
    return client, db


def async_conn_db():
    """
    Establishes an asynchronous connection to the MongoDB server and returns the async client and database objects
    :return:
        async_client: The asynchronous connection to the MongoDB server
        async_db: The database object for the specified file system
    """
    mongo_cluster_uri = _mongo_server_uri()
    async_client = motor.motor_asyncio.AsyncIOMotorClient(mongo_cluster_uri)
    async_db = async_client['filesystem']
    return async_client, async_db


class Users:
    """ User Model """
    def __init__(self, account_name, phone, password, is_admin=False, create_time=None, last_remote_ip=None, is_deleted=False):
        self.account_name = account_name
        self.username = phone
        self.password = password
        self.is_admin = is_admin
        self.last_remote_ip = last_remote_ip
        self.create_time = create_time
        self.is_deleted = is_deleted

    def to_dict(self):
        return {
            "account_name": self.account_name,
            "username": self.username,
            "password": self.password,
            "is_admin": self.is_admin,
            "create_time": self.create_time,
            "is_deleted": self.is_deleted,
        }


class GPUServer:
    def __init__(self, gpu_name, gpu_memory, is_deleted=False):
        self.gpu_name = gpu_name
        self.gpu_memory = gpu_memory
        self.is_deleted = is_deleted

    def to_dict(self):
        return {
            "gpu_name": self.gpu_name,
            "gpu_memory": self.gpu_memory,
            "is_deleted": self.is_deleted,
        }


class Models:
    def __init__(self, model_name, version, desc, created_time, user_id, filename, total_chunk, device, model_source=ModelSource.CUSTOM.value, is_deleted=False):
        self.model_name = model_name
        self.filename = filename
        self.version = version
        self.user_id = user_id
        self.created_time = created_time
        self.desc = desc
        self.device = device
        self.is_deleted = is_deleted
        self.total_chunk = total_chunk
        self.model_source = model_source
        self.upload_status = UploadStatus.IN_PROGRESS.value

    def to_dict(self):
        return {
            "filename": self.filename,
            "model_name": self.model_name,
            "version": self.version,
            "desc": self.desc,
            "user_id": self.user_id,
            "total_chunk": self.total_chunk,
            "created_time": self.created_time,
            "model_source": self.model_source,
            "device": self.device,
            "is_deleted": self.is_deleted,
            "upload_status": self.upload_status,
        }


class Datasets:
    """Dataset Model"""
    def __init__(self, dataset_name, model_alignment, dataset_source, user_id, created_time, filename=None, total_chunk=None, desc=None, is_deleted=False):
        self.dataset_name = dataset_name
        self.model_alignment = model_alignment
        self.dataset_source = dataset_source
        self.filename = filename
        self.user_id = user_id
        self.created_time = created_time
        self.total_chunk = total_chunk
        self.desc = desc
        self.is_deleted = is_deleted
        self.upload_status = UploadStatus.IN_PROGRESS.value

    def to_dict(self):
        return {
            "filename": self.filename,
            "model_name": self.model_alignment,
            "dataset_name": self.dataset_name,
            "dataset_source": self.dataset_source,
            "desc": self.desc,
            "user_id": self.user_id,
            "total_chunk": self.total_chunk,
            "created_time": self.created_time,
            "is_deleted": self.is_deleted,
            "upload_status": self.upload_status,
        }


class DetectionFile:
    """Dataset Model"""
    def __init__(self, filename, task_id, user_id, total_chunk, created_time, is_deleted=False):
        self.filename = filename
        self.task_id = task_id
        self.user_id = user_id
        self.created_time = created_time
        self.is_deleted = is_deleted
        self.total_chunk = total_chunk
        self.upload_status = UploadStatus.IN_PROGRESS.value

    def to_dict(self):
        return {
            "filename": self.filename,
            "task_id": self.task_id,
            "user_id": self.user_id,
            "created_time": self.created_time,
            "is_deleted": self.is_deleted,
            "total_chunk": self.total_chunk,
            "upload_status": self.upload_status,
        }


class DetectionTask:
    def __init__(self, task_id, user_id, model, detect_folder, create_time):
        self.task_id = task_id
        self.user_id = user_id
        self.model = model
        self.create_time = create_time
        self.detect_folder = detect_folder
        self.task_status = TaskStatus.PENDING.value
        self.res = None
        self.is_delete = False

    def to_dict(self):
        return {
            "task_id": self.task_id,
            "user_id": self.user_id,
            "model": self.model,
            "detect_folder": self.detect_folder,
            "task_status": self.task_status,
            "res": self.res,
            "create_time": self.create_time,
            "is_deleted": self.is_delete,
        }


class TrainingTask:
    """TrainingTask Model"""
    def __init__(self, task_id, user_id, model, dataset, epoch, batch_size, learning_rate, create_time):
        self.task_id = task_id
        self.user_id = user_id
        self.model = model
        self.dataset = dataset
        self.epoch = epoch
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.create_time = create_time
        self.task_status = TaskStatus.PENDING.value
        self.res = None
        self.is_deleted = False

    def to_dict(self):
        return {
            "task_id": self.task_id,
            "user_id": self.user_id,
            "model": self.model,
            "dataset": self.dataset,
            "epoch": self.epoch,
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
            "create_time": self.create_time,
            "task_status": self.task_status,
            "res": self.res,
            "is_deleted": self.is_deleted,
        }


def init_database():
    """
    Initializes the database by creating indexes on relevant collections

    """
    client, db = conn_db()

    training_task_collection = db['TrainingTask']
    training_task_collection.create_index([('task_id', ASCENDING)], unique=True)

    detect_task_collection = db['DetectionTask']
    detect_task_collection.create_index([('task_id', ASCENDING)], unique=True)

    models_collection = db['Models']
    datasets_collection = db['Datasets']
    detection_files_collection = db['DetectionFiles']
    gpu_collection = db['GPUServerInfos']

    client.close()


if __name__ == "__main__":
    init_database()