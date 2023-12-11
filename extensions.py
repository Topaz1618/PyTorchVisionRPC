import uuid
from datetime import datetime

from gridfs import GridFS
import motor.motor_asyncio


from mongodb_models import (conn_db, async_conn_db, DetectionTask, TrainingTask, DetectionFile, Models, Datasets,
                            GPUServer, Users)
from enums import TaskStatus, UploadStatus, ModelSource, DatasetSource


class MongoDBManager:
    """
    A manager class for interacting with the MongoDB database.
    """
    def __init__(self):
        self.client, self.db = conn_db()
        self.async_client, self.async_db = async_conn_db()
#         self.fs = motor.motor_asyncio.AsyncIOMotorGridFSBucket(self.async_db)
        self.gridfs = GridFS(self.db, collection='fs')
        self.gridfs_collection = self.db['fs']

        self.users_collection = self.db['AIUsers']
        self.detect_task_collection = self.db['DetectionTask']
        self.training_task_collection = self.db['TrainingTask']
        self.datasets_collection = self.db['Datasets']
        self.models_collection = self.db['Models']
        self.detection_files_collection = self.db['DetectionFiles']
        self.gpu_collection = self.db['GPUServerInfos']


# class AsyncGridFSManager(MongoDBManager):
#     def __init__(self):
#         super().__init__()

#     async def upload_chunk(self, data, filename):
#         upload_stream = self.fs.open_upload_stream(filename=filename)
#         try:
#             await upload_stream.write(data)

#         finally:
#             await upload_stream.close()
#         print(f'Uploaded file {filename} to GridFS')

#     async def count_file_chunks(self, filename):
#         clip_count = await self.fs.find({'filename': filename}).to_list(length=None)
#         return clip_count

#     async def download_chunk(self, chunk_id):
#         download_stream = await self.fs.open_download_stream(chunk_id)
#         return download_stream

#     async def delete_files_async(self, filename):
#         versions = await self.fs.find({'filename': filename}).to_list(length=None)

#         for version in versions:
#             chunk_id = version['_id']
#             await self.fs.delete(chunk_id)

#     def close(self):
#         self.client.close()


class DetectionTaskManager(MongoDBManager):
    """
    A manager class for user-related operations in MongoDB.
    Inherits from MongoDBManager.
    """

    def __init__(self):
        super().__init__()

    def create_task(self, task_id, user_id, model, detect_folder, create_time):
        task_info = DetectionTask(task_id, user_id, model, detect_folder, create_time)
        task_dict = task_info.to_dict()
        print(task_dict)
        self.detect_task_collection.insert_one(task_dict)

    def get_tasks_count(self, user, is_admin, task_status=None):
        query = {"is_deleted": False}

        if task_status is not None:
            query["task_status"] = task_status

        if not is_admin:
            query["user_id"] = user

        total_count = self.detect_task_collection.count_documents(query)
        return total_count

    def get_tasks(self, start, end, user, is_admin):
        query = {"is_deleted": False}

        if not is_admin:
            query["user_id"] = user

        tasks = self.detect_task_collection.find(query).skip(start).limit(end - start)

        all_task_list = list()
        for task in tasks:
            task["_id"] = str(task.get("_id"))
            all_task_list.append(task)

        return all_task_list

    def get_task(self, task_id):
        task_info = self.detect_task_collection.find_one({"task_id": task_id})
        return task_info

    def delete_task(self, task_id):
        self.detect_task_collection.update_one({"task_id": task_id}, {"$set": {"is_deleted": True}})

    def update_task(self, task_id, key, value):
        updated_fields = {key: value}
        self.detect_task_collection.update_one({"task_id": task_id}, {"$set": updated_fields})

    def close(self):
        self.client.close()

class TrainingTaskManager(MongoDBManager):
    def __init__(self):
        super().__init__()

    def create_task(self, task_id, user_id, model, dataset, epoch, batch_size,
                                 learning_rate, create_time):

        task_info = TrainingTask(task_id, user_id, model, dataset, epoch, batch_size,
                                 learning_rate, create_time)
        self.training_task_collection.insert_one(task_info.to_dict())

    def get_tasks_count(self, user, is_admin, task_status=None):
        query = {"is_deleted": False}

        if task_status is not None:
            query["task_status"] = task_status

        if not is_admin:
            query["user_id"] = user

        total_count = self.training_task_collection.count_documents(query)
        return total_count

    def get_tasks(self, start, end, user, is_admin):
        query = {"is_deleted": False}

        if not is_admin:
            query["user_id"] = user

        tasks = self.detect_task_collection.find(query).skip(start).limit(end - start)

        all_task_list = list()
        for task in tasks:
            task["_id"] = str(task.get("_id"))
            all_task_list.append(task)

        return all_task_list

    def get_task(self, task_id):
        task_info = self.training_task_collection.find_one({"task_id": task_id})
        return task_info

    def delete_task(self, task_id):
        self.training_task_collection.update_one({"task_id": task_id}, {"$set": {"is_deleted": True}})

    def update_task(self, task_id, key, value):
        updated_fields = {key: value}
        self.training_task_collection.update_one({"task_id": task_id}, {"$set": updated_fields})

    def close(self):
        self.client.close()


class ModelsManager(MongoDBManager):
    def __init__(self):
        super().__init__()

    def create_model(self, model_name, version, desc, created_time, user_id,  filename=None, total_chunk=None,
                     device=None, model_source=ModelSource.CUSTOM.value):
        model_info = Models(model_name, version, desc, created_time, user_id, filename, total_chunk, device, model_source)
        self.models_collection.insert_one(model_info.to_dict())

    def get_models_count(self, user, is_admin):
        query = {"is_deleted": False}

        if not is_admin:
            query["user_id"] = user

        total_count = self.models_collection.count_documents(query)
        return total_count

    def get_models(self, start, end, user, is_admin):
        query = {"is_deleted": False, "upload_status": UploadStatus.COMPLETED.value}

        if not is_admin:
            query["user_id"] = user

            new_query = {"$or": [
                query,
                {"version": "default"}
            ]}

            query = new_query

        models = self.models_collection.find(query).skip(start).limit(end - start)

        all_models_list = list()
        for model in models:
            model["_id"] = str(model.get("_id"))
            all_models_list.append(model)

        return all_models_list

    def get_model(self, model_name, version):
        task_info = self.models_collection.find_one({"model_name": model_name, "version": version})
        return task_info

    def delete_model(self, model_name, version, filename):
        self.models_collection.update_one({"model_name": model_name, "version": version, "filename": filename}, {"$set": {"is_deleted": True}})

    def update_model(self, model_name, version, key, value):
        updated_fields = {key: value}
        self.models_collection.update_one({"model_name": model_name, "version": version}, {"$set": updated_fields})

    def is_exists(self, model_name, version=None, filename=None, username=None):
        query = {"model_name": model_name, "filename": filename, "version": version, "is_deleted": False, "upload_status": UploadStatus.COMPLETED.value}

        model = self.models_collection.find_one(query)

        if model is None:
            return False
        else:
            return True

    def close(self):
        self.client.close()


class UserManager(MongoDBManager):
    def __init__(self):
        super().__init__()

    def create_user(self, username, phone, password, is_admin, created_time):
        user_info = Users(username, phone, password, is_admin, created_time)
        self.users_collection.insert_one(user_info.to_dict())

    def get_users_count(self):
        query = {"is_deleted": False}
        total_count = self.users_collection.count_documents(query)
        return total_count

    def get_users(self, start, end):
        query = {"is_deleted": False}

        users = self.users_collection.find(query).skip(start).limit(end - start)

        all_users_list = list()
        for user in users:
            user["_id"] = str(user.get("_id"))
            all_users_list.append(user)

        return all_users_list

    def get_user(self, username):

        task_info = self.users_collection.find_one({"username": username})
        return task_info

    def delete_user(self, username):
        self.users_collection.update_one({"username": username}, {"$set": {"is_deleted": True}})

    def update_user(self, username, key, value):
        updated_fields = {key: value}
        self.users_collection.update_one({"username": username}, {"$set": updated_fields})

    def is_exists(self, username):
        print("is exists", username)
        query = {"username": username, "is_deleted": False}

        user = self.users_collection.find_one(query)
        print(user)

        if user is None:
            return False
        else:
            return True

    def check_is_admin(self, username):
        query = {'username': username}
        user = self.users_collection.find_one(query)
        if user:
            is_admin = user.get("is_admin")
            return is_admin
        return False

    def check_user_password(self, user_doc, password):
        if user_doc.get("password") == password:
            return True
        return False

    def update_password(self, username, password):
        self.users_collection.update_one(
            {'username': username},
            {'$set': {'password': password}}
        )

    def update_remote_ip(self, username, remote_ip):
        self.users_collection.update_one(
            {'username': username},
            {'$set': {'last_remote_ip': remote_ip}}
        )

    def close(self):
        self.client.close()


class DatasetsManager(MongoDBManager):
    def __init__(self):
        super().__init__()

    def create_dataset(self, dataset_name, model_alignment, dataset_source, user_id, created_time, filename=None, total_chunk=None, desc=None):
        dataset_info = Datasets(dataset_name, model_alignment, dataset_source, user_id, created_time, filename, total_chunk, desc)
        self.datasets_collection.insert_one(dataset_info.to_dict())

    def get_datasets_count(self, user, is_admin):
        query = {"is_deleted": False}

        if not is_admin:
            query["user_id"] = user

        total_count = self.datasets_collection.count_documents(query)
        return total_count

    def get_datasets(self, start, end, user, is_admin):
        query = {"is_deleted": False, "upload_status": UploadStatus.COMPLETED.value}

        if not is_admin:
            query["user_id"] = user

        datasets = self.datasets_collection.find(query).skip(start).limit(end - start)

        all_datasets_list = list()
        for dataset in datasets:
            dataset["_id"] = str(dataset.get("_id"))
            all_datasets_list.append(dataset)

        return all_datasets_list

    def get_model_assoite_datasets(self, start, end, user, is_admin):
        query = {"is_deleted": False, "upload_status": UploadStatus.COMPLETED.value}

        if not is_admin:
            query["user_id"] = user

            new_query = {"$or": [
                query,
                {"dataset_source": DatasetSource.BUILT_IN.value},
            ]}

            query = new_query

        if not is_admin:
            query["user_id"] = user

        datasets = self.datasets_collection.find(query).skip(start).limit(end - start)

        all_datasets_list = list()
        for dataset in datasets:
            dataset["_id"] = str(dataset.get("_id"))
            all_datasets_list.append(dataset)

        return all_datasets_list

    def get_dataset(self, dataset_name):
        dataset_info = self.datasets_collection.find_one({"dataset_name": dataset_name})
        return dataset_info

    def delete_dataset(self, dataset_name, filename):
        self.datasets_collection.update_one({"dataset_name": dataset_name, "filename": filename}, {"$set": {"is_deleted": True}})

    def update_dataset(self, dataset_name, key, value):
        updated_fields = {key: value}
        self.datasets_collection.update_one({"dataset_name": dataset_name}, {"$set": updated_fields})

    def is_exists(self, dataset_name, user=None):
        query = {"dataset_name": dataset_name, "is_deleted": False, "upload_status": UploadStatus.COMPLETED.value}
        if user:
            query["user_id"] = user

        dataset = self.datasets_collection.find_one(query)
        if dataset is None:
            return False
        else:
            return True

    def close(self):
        self.client.close()

class GPUServerManager(MongoDBManager):
    def __init__(self):
        super().__init__()

    def create_gpu_server(self, gpu_name, memory):
        gpu_info = GPUServer(gpu_name, memory)
        self.gpu_collection.insert_one(gpu_info.to_dict())

    def get_gpu_server_count(self, user, is_admin):
        query = {"is_deleted": False}
        total_count = self.gpu_collection.count_documents(query)
        return total_count

    def get_gpu_servers(self, start, end, user, is_admin):
        query = {"is_deleted": False}

        gpus = self.gpu_collection.find(query).skip(start).limit(end - start)

        all_servers_list = list()
        for gpu_info in gpus:
            gpu_info["_id"] = str(gpu_info.get("_id"))
            all_servers_list.append(gpu_info)

        return all_servers_list

    def get_gpu(self, gpu_name, gpu_memory):
        gpu_info = self.gpu_collection.find_one({"gpu_name": gpu_name, "gpu_memory": gpu_memory})
        return gpu_info

    def delete_gpu(self, gpu_name):
        self.gpu_collection.update_one({"gpu_name": gpu_name}, {"$set": {"is_deleted": True}})

    def update_gpu(self, gpu_name, key, value):
        updated_fields = {key: value}
        self.gpu_collection.update_one({"gpu_name": gpu_name}, {"$set": updated_fields})

    def is_exists(self, gpu_name, gpu_memory):
        query = {"gpu_name": gpu_name, "gpu_memory": gpu_memory, "is_deleted": False }

        gpu_info = self.gpu_collection.find_one(query)
        if gpu_info is None:
            return False
        else:
            return True

    def close(self):
        self.client.close()


class DetectionFileManager(MongoDBManager):
    def __init__(self):
        super().__init__()

    def create_file(self, filename, task_id, user_id, total_chunk, created_time):
        file_info = DetectionFile(filename, task_id, user_id, total_chunk, created_time)
        self.detection_files_collection.insert_one(file_info.to_dict())

    def get_files_count(self, user, is_admin):
        query = {"is_deleted": False}

        if not is_admin:
            query["user_id"] = user

        total_count = self.detection_files_collection.count_documents(query)
        return total_count

    def get_files(self, start, end, user, is_admin):
        query = {"is_deleted": False}

        if not is_admin:
            query["user_id"] = user

        files = self.detection_files_collection.find(query).skip(start).limit(end - start)

        all_files_list = list()
        for file in files:
            file["_id"] = str(file.get("_id"))
            all_files_list.append(file)

        return all_files_list

    def get_file(self, filename, task_id):
        file_info = self.detection_files_collection.find_one({"filename": filename, "task_id": task_id})
        return file_info

    def delete_file(self, filename, task_id):
        self.detection_files_collection.update_one({"filename": filename, "task_id": task_id}, {"$set": {"is_deleted": True}})

    def update_file_status(self, filename, task_id, key, value):
        updated_fields = {key: value}
        self.detection_files_collection.update_one({"filename": filename, "task_id": task_id}, {"$set": updated_fields})

    def is_exists(self, filename, user=None):
        query = {"filename": filename, "is_deleted": False, "upload_status": UploadStatus.COMPLETED.value}
        if user:
            query["user_id"] = user

        dataset = self.detection_files_collection.find_one(query)
        if dataset is None:
            return False
        else:
            return True

    def close(self):
        self.client.close()


if __name__ == "__main__":
    # Detection task

    task_id = str(uuid.uuid4())  # 生成唯一的任务ID
    create_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    user_id = "Topaz"
    model = "Tesseract"
    detect_folder = 'detect_demo1.zip'

    detection_task_obj = DetectionTaskManager()
    # detection_task_obj.create_task(task_id, user_id, model, detect_folder, create_time)

    task_id = '09cd7752-c683-4802-bc63-272ac5a3025a'
    print(detection_task_obj.get_task(task_id))

    # detection_task_obj.update_task(task_id, "status", TaskStatus.CANCELLED.value)
    # print(detection_task_obj.get_task(task_id))

    # print(detection_task_obj.get_tasks_count(user_id, is_admin=False))


    # start, end, user, is_admin
    # print(detection_task_obj.get_tasks(0, 4, user_id, is_admin=False))

