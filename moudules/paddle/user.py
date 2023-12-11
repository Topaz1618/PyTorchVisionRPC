from base import BaseHandler

from extensions import (DatasetsManager, ModelsManager, TrainingTaskManager, DetectionTaskManager,
                        AsyncGridFSManager)

from file_utils import generate_dataset_gridfs_save_name, generate_model_gridfs_save_name

class UserDatasetListHandler(BaseHandler):
    async def get(self):
        # 处理获取数据集管理的逻辑
        username = "Topaz"
        is_admin = True
        dataset_obj = DatasetsManager()
        datasets_count = dataset_obj.get_datasets_count(username, is_admin)
        data_list = dataset_obj.get_datasets(0, datasets_count, username, is_admin)
        print(data_list)
        # self.write({"msg":data_list })
        self.render("user_datasets.html", data=data_list)


class UserModelListHandler(BaseHandler):
    def get(self):
        username = "Topaz"
        is_admin = True
        model_obj = ModelsManager()
        models_count = model_obj.get_models_count(username, is_admin)
        data_list = model_obj.get_models(0, models_count, username, is_admin)
        print(data_list)
        self.render("user_models.html", data=data_list)


class DeleteUserDatasetHandler(BaseHandler):
    def get(self):
        # 处理获取数据集管理的逻辑
        pass

    def post(self):
        # 处理创建新数据集的逻辑
        pass



class DeleteUserModelHandler(BaseHandler):
    def get(self):
        # 处理获取数据集管理的逻辑
        pass

    def post(self):
        # 处理创建新数据集的逻辑
        pass


class UserTaskListHandler(BaseHandler):
    def get(self):
        # 处理获取数据集管理的逻辑
        pass

    def post(self):
        # 处理创建新数据集的逻辑
        pass


class DeleteUserTaskHandler(BaseHandler):
    def get(self):
        # 处理获取数据集管理的逻辑
        pass

    def post(self):
        # 处理创建新数据集的逻辑
        pass


class SingleUserTaskHandler(BaseHandler):
    def get(self):
        # 处理获取数据集管理的逻辑
        pass

    def post(self):
        # 处理创建新数据集的逻辑
        pass


class DownloadDatasetHandler(BaseHandler):
    async def get(self):
        # 处理获取数据集管理的逻辑
        gridfs_manager = AsyncGridFSManager()

        chunk_number = int(self.request.headers.get('X-Chunk-Number', default='0'))
        dataset_name = self.get_argument("dataset_name", None)
        username = self.get_argument("username", None)

        self.set_header('Content-Type', 'application/octet-stream')
        gridfs_save_name = generate_dataset_gridfs_save_name(dataset_name, username)

        versions = await gridfs_manager.count_file_chunks(gridfs_save_name)
        version = versions[chunk_number]  # 指定切片
        chunk_id = version['_id']

        download_stream = await gridfs_manager.download_chunk(chunk_id)

        data = await download_stream.read()
        self.write(data)

    def post(self):
        # 处理创建新数据集的逻辑
        pass


class DownloadModelHandler(BaseHandler):
    async def get(self):
        # 处理获取数据集管理的逻辑
        chunk_number = int(self.request.headers.get('X-Chunk-Number', default='0'))
        model_name = self.get_argument("model_name", None)
        version = self.get_argument("version", None)
        username = self.get_argument("username", None)

        gridfs_manager = AsyncGridFSManager()

        self.set_header('Content-Type', 'application/octet-stream')
        gridfs_save_name = generate_model_gridfs_save_name(model_name, version, username)

        versions = await gridfs_manager.count_file_chunks(gridfs_save_name)
        version = versions[chunk_number]  # 指定切片
        chunk_id = version['_id']

        download_stream = await gridfs_manager.download_chunk(chunk_id)

        data = await download_stream.read()
        self.write(data)

    def post(self):
        # 处理创建新数据集的逻辑
        pass


class DownloadDetectionFileHandler(BaseHandler):
    def get(self):
        # 处理获取数据集管理的逻辑
        pass

    def post(self):
        # 处理创建新数据集的逻辑
        pass