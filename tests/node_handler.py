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

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

from enums import TaskType, TaskStatus, WorkerConnectionStatus
from task_utils import (get_task_from_detect_queue, remove_task_from_detect_queue, create_task, generate_detect_task_command,
                        generate_training_task_command, get_task_info, update_task_info, generate_task_command, delete_task_key)

detect_nodes = {
    "detect_node1": {
        "ip": "127.0.0.1",
        "capacity": 1,  # 初始能力数值为1
    },
    # "detect_node2": {
    #     "ip": "127.0.0.1",
    #     "capacity": 1,  # 初始能力数值为1
    # },
}


class TaskUpdateType(Enum):
    ADD_NODE = 1
    STOP = 2


redis_client = redis.Redis(host="127.0.0.1", port=6379)


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

    def get_detect_command(self):
        # # 从任务队列中获取并处理任务
        command = generate_detect_task_command()
        print(command)
        return command

    def get_training_command(self):
        # # 从任务队列中获取并处理任务
        command = generate_training_task_command()
        print(command)

        return command

    def create_detect_task(self):
        # 创建识别任务
        task_id = str(uuid.uuid4())  # 生成唯一的任务ID
        create_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        task = create_task(task_id, TaskType.DETECT.value, create_time, 'tablenet', 'detect_demo1.zip')
        print("Task info", task)

        print(f"Current detect task:  {self.get_detect_task_queue()} \n")

        self.add_task_to_detect_queue(task_id)

        print(f"Current detect task: {self.get_detect_task_queue()} \n")
        return task_id

    def create_training_task(self):
        # 创建训练任务
        task_id = str(uuid.uuid4())  # 生成唯一的任务ID
        create_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        task_info = create_task(task_id, TaskType.TRAINING.value, create_time, 'tablenet', 'WWT', 10, 'random', 32, 0.001, 'ReLU')
        print("Task info", task_info)

        print(f"Current detect task:  {self.get_training_task_queue()} \n")

        self.add_task_to_training_queue(task_id)

        print(f"Current detect task:  {self.get_training_task_queue()} \n")
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

    # def generate_task_command(self, task_type):
    #     res = generate_task_command(task_type)
    #     return res

    def get_all_tasks(self):
        res = redis_client.hgetall('task_dict')
        print(res)

    def delete_task_key(self, task_id, key):
        delete_task_key(task_id, key)


def get_idle_workers(used_capacity, total_nodes):
    used_capacity = used_capacity if used_capacity else {}

    if not used_capacity:
        for node, capacity_info in total_nodes.items():
            if capacity_info["capacity"] > 0:
                return node

    for node, capacity in total_nodes.items():
        used_capacity = used_capacity.get(node.encode(), {})
        if isinstance(used_capacity, bytes):
            used_capacity = used_capacity.decode()

        if isinstance(used_capacity, str):
            used_capacity = json.loads(used_capacity)

        used_capacity_num = used_capacity.get("cap_num", 0)
        print("used_capacity_num", used_capacity_num)
        available_capacity_num = capacity["capacity"] - used_capacity_num

        if available_capacity_num > 0:
            return node

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


def add_used_capacity(node):
    if not isinstance(node, bytes):
        node = node.encode()

    if redis_client.hexists('used_detect_nodes', node):
        existing_value = redis_client.hget("used_detect_nodes", node)
        print(existing_value)
        existing_value_dict = json.loads(existing_value.decode('utf-8'))
        # 检查字典中是否有'cap_num'键
        if 'cap_num' in existing_value_dict:
            # 如果有'cap_num'键，将其值加1
            existing_value_dict['cap_num'] += 1
        else:
            # 如果没有'cap_num'键，创建'cap_num'键并设置为1
            existing_value_dict['cap_num'] = 1

        redis_client.hset('used_detect_nodes', node, json.dumps(existing_value_dict))

        print(redis_client.hgetall('used_detect_nodes'))
    else:

        redis_client.hset('used_detect_nodes', node, json.dumps({'cap_num': 1}))


def remove_used_capacity(node, used_capacity):
    if not isinstance(node, bytes):
        node = node.encode()

    if redis_client.hexists('used_detect_nodes', node):
        existing_value = redis_client.hget("used_detect_nodes", node)
        print(existing_value)

        existing_value_dict = json.loads(existing_value.decode('utf-8'))
        # 检查字典中是否有'cap_num'键
        if 'cap_num' in existing_value_dict:
            # 如果有'cap_num'键，将其值加1
            existing_value_dict['cap_num'] -=1
        else:
            # 如果没有'cap_num'键，创建'cap_num'键并设置为1
            print("There's no key called cap_num")

        redis_client.hset('used_detect_nodes', node, json.dumps(existing_value_dict))

        print(redis_client.hgetall('used_detect_nodes'))
    else:
        print("节点不存在")


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
                command = generate_task_command(task_id, TaskType.DETECT.value)
                print(f"Gonna start process: {command}")
                command = command.split()
                pipe = sp.Popen(command, stderr=sp.STDOUT, universal_newlines=True)

                print("Task PID", pipe.pid)
                # add_used_capacity(idle_detect_node)
                #
                #
                # remove_task_from_detect_queue(task_id)

            else:
                print("no task in the detect queue")
                continue

            # add_used_capacity(idle_detect_node)



            # worker_dic 中更新使用的 node



        """
        后台监听空闲节点
            1. 节点空闲就获取任务 
            2. 获取任务相关信息，启动进程
                - 队列中移除任务
                - worker_dic 中更新使用的 node
                - 增加 node 到 used_node
            3. 任务完成
                - 移除 node 到 used_node
                - 更新任务状态为已完成
                
        """

        # if idle_detect_node:
        #     data = generate_task_command(TaskType.DETECT.value)
        #     print("Data", data)
        #
        #     task_id, command = data
        #     task_info = {"task_id": task_id, "command": command}



if __name__ == "__main__":
    t = threading.Thread(target=queue_listener)
    t.start()
    used_nodes = redis_client.hgetall('used_detect_nodes')
    print("Detect", used_nodes)

    task_obj = TaskHandlerDemo()
    print(f"Detect Queue: {task_obj.get_detect_task_queue()}")

    detect_task_id = task_obj.create_detect_task()
    # print(f"Command: {task_obj.generate_task_command(TaskType.DETECT.value)}")
    # print(f"Detect Queue: {task_obj.get_detect_task_queue()}")

    # 测试用例
    # used_capacity = {b'worker1': b'{"cap_num": 1}', b'detect_node1': b'{"cap_num": 1}'}
    # node = 'worker2'

    # update_used_capacity_local(used_capacity, node)
    # print(used_capacity)

    node = 'worker3'

    """
         - API 创建任务
            1. 任务增加到 task
            2. 任务增加到待处理队列
            
        - 后台监听空闲节点
            1. 节点空闲就获取任务
            2. 获取任务相关信息，启动进程
                - 队列中移除任务
                - worker_dic 中更新使用的 node
                - 增加 node 到 used_node
            3. 任务完成
                - 移除 node 到 used_node
                - 更新任务状态为已完成
                
            
    """


    # # 增加使用中节点
    # add_used_capacity(node, used_nodes)
    #
    # used_nodes = redis_client.hgetall('used_detect_nodes')
    # print("Detect", used_nodes)
    #
    # # Todo: 处理任务
    #
    #
    # remove_used_capacity(node, used_nodes)
    # print("移除使用中节点")