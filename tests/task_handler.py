import os
import sys
import uuid
import redis
import json

from time import time, sleep
from datetime import datetime

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

from enums import TaskType, TaskStatus, WorkerConnectionStatus
from task_utils import (add_training_task_to_queue, add_detect_task_to_queue, create_task, generate_detect_task_command,
                        generate_training_task_command, get_task_info, update_task_info, generate_task_command, delete_task_key)


redis_client = redis.Redis(host='localhost', port=6379)


# 示例, 测试使用方法
class TaskHandlerDemo:
    def get_detect_tasks(self):
        detect_task_queue = redis_client.lrange("detect_task_queue", 0, -1)
        return detect_task_queue

    def get_training_tasks(self):
        training_task_queue = redis_client.lrange("training_task_queue", 0, -1)
        return training_task_queue

    def create_detect_task(self):
        # 创建识别任务
        task_id = str(uuid.uuid4())  # 生成唯一的任务ID
        create_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        task = create_task(task_id, TaskType.DETECT.value, create_time, 'tablenet', 'detect_demo1.zip')
        print("Task info", task)

        print(f"Current detect task:  {self.get_detect_tasks()} \n")

        add_detect_task_to_queue(task_id)

        print(f"Current detect task: {self.get_detect_tasks()} \n")
        return task_id

    def get_detect_command(self, task_id):
        # # 从任务队列中获取并处理任务
        command = generate_detect_task_command(task_id)
        print(command)
        return command


    def create_training_task(self):
        # 创建训练任务
        task_id = str(uuid.uuid4())  # 生成唯一的任务ID
        create_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        task_info = create_task(task_id, TaskType.TRAINING.value, create_time, 'tablenet', 'WWT', 10, 'random', 32, 0.001, 'ReLU')
        print("Task info", task_info)

        print(f"Current detect task:  {self.get_training_tasks()} \n")

        add_training_task_to_queue(task_id)

        print(f"Current detect task:  {self.get_training_tasks()} \n")
        return task_id

    def get_training_command(self):
        # # 从任务队列中获取并处理任务
        command = generate_training_task_command()
        print(command)

        return command

    def sub_demo(self):
        # 订阅发布机制(弃用)
        redis_client.publish('queue_notification_channel', 'New Task')

    def get_task_info(self, task_id):
        task_info = get_task_info(task_id)
        print(task_info)

        return task_info

    def update_task_info(self, task_id):
        update_task_info(task_id, "progress", "20")
        print(self.get_task_info(task_id))

        for i in range(10):
            update_task_info(task_id, "progress", i)
            print(self.get_task_info(task_id))


    def get_all_tasks(self):
        res = redis_client.hgetall('task_dict')
        print(res)

    def delete_task_key(self, task_id, key):
        delete_task_key(task_id, key)


if __name__ == "__main__":
    task_obj = TaskHandlerDemo()
    # detect_task_id = task_obj.create_detect_task()
    # print(f"Command: {task_obj.generate_task_command(TaskType.DETECT.value)}")

    detect_task_id = "4ba5ff7d-f43c-4676-9dd8-57dc4339f1d6"
    task_obj.get_task_info(detect_task_id)
    # update_task_info(detect_task_id, "status", 2)
    # delete_task_key(detect_task_id, "log")
    # task_obj.get_task_info(detect_task_id)
    # task_obj.get_all_tasks()

    # training_task_id = task_obj.create_training_task()
    # task_obj.get_task_info(training_task_id.encode())
    # print(f"Command: {task_obj.generate_task_command(TaskType.TRAINING.value)}")
    # print(task_obj.get_detect_tasks())

