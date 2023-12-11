import os
import sys
import threading
import uuid
import subprocess as sp
import redis
import json

import json
import redis
from enum import Enum

from time import time, sleep
from datetime import datetime

# parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# sys.path.append(parent_dir)

from enums import TaskType, TaskStatus, WorkerConnectionStatus, DetectionAlgorithm, TrainModelType
from task_utils import (get_task_from_detect_queue, remove_task_from_detect_queue, create_task, get_idle_workers,
                        generate_training_task_command, get_task_info, update_task_info, generate_task_command, delete_task_key,
                        add_used_capacity, remove_task_from_training_queue, get_task_from_training_queue, remove_used_capacity)
from config import USED_TRAINING_NODES, USED_DETECT_NODES, detect_nodes, training_nodes, REDIS_PORT, REDIS_HOST



class TaskUpdateType(Enum):
    ADD_NODE = 1
    STOP = 2


class ErrorKeyword(Enum):
    ERROR = "Error"
    ERR = "Err"
    RAISE = "raise"
    ERROR_CASE_INSENSITIVE = "error"
    # TRACEBACK = "Traceback"


redis_client = redis.Redis(host=REDIS_HOST, port=REDIS_PORT)


class TaskHandlerDemo:
    def add_task_to_training_queue(self, task_id):
        redis_client.rpush('training_task_queue', task_id)

    def add_task_to_detect_queue(self, task_id):
        redis_client.rpush('detect_task_queue', task_id)

    def get_detect_task_queue(self):
        detect_task_queue = redis_client.lrange("detect_task_queue", 0, -1)
        return detect_task_queue

    def get_training_task_queue(self):
        training_task_queue = redis_client.lrange("training_task_queue", 0, -1)
        return training_task_queue

    def create_detect_task(self):
        # 创建识别任务
        task_id = str(uuid.uuid4())  # 生成唯一的任务ID
        create_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        task = create_task(task_id, TaskType.DETECT.value, create_time, DetectionAlgorithm.TESSERACT.value , 'detect_demo1.zip')
        print("Task info", task)

        print(f"Current detect task:  {self.get_detect_task_queue()} \n")

        self.add_task_to_detect_queue(task_id)

        print(f"Current detect task: {self.get_detect_task_queue()} \n")
        return task_id

    def create_training_task(self):
        # 创建训练任务1
        task_id = str(uuid.uuid4())  # 生成唯一的任务ID
        create_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        task_info = create_task(task_id, TaskType.TRAINING.value, create_time, TrainModelType.YOLO.value, 'WWT', 10, 1, 32, 0.001, 'ReLU')
        print("Task info", task_info)

        print(f"Current training task:  {self.get_training_task_queue()} \n")

        self.add_task_to_training_queue(task_id)

        print(f"Current training task:  {self.get_training_task_queue()} \n")
        return task_id

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


def update_used_capacity_local(used_capacity, node):

    if not isinstance(node, bytes):
        node = node.encode()

    if node in used_capacity:
        used_capacity_node = used_capacity[node]
        if isinstance(used_capacity_node, bytes):
            used_capacity_node = used_capacity_node.decode()

        if isinstance(used_capacity_node, str):
            used_capacity_node = json.loads(used_capacity_node)

        used_capacity_node["cap_num"] = used_capacity_node.get("cap_num", 0) + 1
        used_capacity[node] = json.dumps(used_capacity_node)
    else:
        used_capacity[node] = json.dumps({"cap_num": 1})


def monitor_detect_sub_process(pipe, node, task_id):
    try:
        while True:
            if pipe is None:
                break

            data = pipe.stdout.readline().strip()
            if data:

                if any(keyword.value in data for keyword in ErrorKeyword):
                    raise Exception(data)

                else:
                    print(f"[output]:", data)

    except Exception as e:
        print("Error occurred in subprocess:", e)
        # raise e
        update_task_info(task_id, "log", f"[Error] : {e}")
        update_task_info(task_id, "status", TaskStatus.FAILED.value)
        remove_used_capacity(node, TaskType.DETECT.value)


def queue_listener():
    while True:

        sleep(2)
        # 目前如果已使用为空，可以直接返回节点
        used_nodes = redis_client.hgetall('used_detect_nodes')
        print("Detect node in using:", used_nodes)

        idle_detect_node = get_idle_workers(used_nodes, detect_nodes)
        print("Idle detect node", idle_detect_node)

        if idle_detect_node:
            # 增加 node 到 used_node
            task_id = get_task_from_detect_queue()
            if task_id:
                command = generate_task_command(task_id, idle_detect_node, TaskType.DETECT.value)
                # command = f"python detect_demo.py --task_id {task_id} --model {DetectionAlgorithm.TESSERACT.value}"

                print(f"Gonna start process: {command}")
                try:
                    command = command.split()
                    pipe = sp.Popen(command, stdout=sp.PIPE, stderr=sp.STDOUT,
                                    universal_newlines=True)

                    # pipe = sp.Popen(command, stderr=sp.STDOUT, universal_newlines=True)
                    #
                    print("Task process pid: ", pipe.pid)
                    monitor_thread = threading.Thread(target=monitor_detect_sub_process, args=(pipe, idle_detect_node, task_id))
                    monitor_thread.start()

                    print("Task PID", pipe.pid)
                    update_task_info(task_id, "task_pid", pipe.pid)

                except Exception as e:
                    print("Start sub process error", e)

                used_nodes = redis_client.hgetall('used_detect_nodes')
                print("Before Detect", used_nodes)
                add_used_capacity(idle_detect_node, TaskType.DETECT.value)
                used_nodes = redis_client.hgetall('used_detect_nodes')
                print("After Detect ", used_nodes)

                remove_task_from_detect_queue(task_id)
                print(get_task_from_detect_queue())
            else:
                print("no task in the detect queue")
                continue


def training_queue_listener():
    while True:

        sleep(2)
        # 目前如果已使用为空，可以直接返回节点
        used_nodes = redis_client.hgetall(USED_TRAINING_NODES)
        print("Training node in using:", used_nodes)

        idle_training_node = get_idle_workers(used_nodes, training_nodes)
        print("Idle training node", idle_training_node)

        if idle_training_node:

            # 增加 node 到 used_node
            task_id = get_task_from_training_queue()
            if task_id:
                command = generate_task_command(task_id, idle_training_node, TaskType.TRAINING.value)
                print(f"Gonna start process: {command}")
                command = command.split()
                pipe = sp.Popen(command, stderr=sp.STDOUT, universal_newlines=True)

                print("Task PID", pipe.pid)
                update_task_info(task_id, "task_pid", pipe.pid)

                used_nodes = redis_client.hgetall(USED_TRAINING_NODES)
                print("Before Detect", used_nodes)
                add_used_capacity(idle_training_node, TaskType.TRAINING.value)
                used_nodes = redis_client.hgetall(USED_TRAINING_NODES)
                print("After Detect ", used_nodes)

                remove_task_from_training_queue(task_id)
                print(get_task_from_training_queue())
            else:
                print("no task in the detect queue")
                continue


if __name__ == "__main__":
    # t = threading.Thread(target=queue_listener)
    # t.start()

    task_obj = TaskHandlerDemo()
    print(f"Detect Queue: {task_obj.get_detect_task_queue()}")

    used_nodes = redis_client.hgetall('used_detect_nodes')
    print("Detect", used_nodes)

    detect_task_id = task_obj.create_detect_task()
    detect_task_id2 = task_obj.create_detect_task()

    print(f"Detect Queue: {task_obj.get_detect_task_queue()}")

    # t = threading.Thread(target=training_queue_listener)
    # t.start()
    # used_nodes = redis_client.hgetall(USED_TRAINING_NODES)
    # print("Training bust nodes: ", used_nodes)
    # print(f"Training Queue: {task_obj.get_training_task_queue()}")
    # training_task_id = task_obj.create_training_task()

    # 测试用例
    # used_capacity = {b'worker1': b'{"cap_num": 1}', b'detect_node1': b'{"cap_num": 1}'}
    # node = 'worker2'

    # update_used_capacity_local(used_capacity, node)
    # print(used_capacity)

