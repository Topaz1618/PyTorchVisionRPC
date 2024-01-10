import os
import json
import asyncio
import threading
import subprocess as sp
from datetime import datetime
from time import sleep, time
import multiprocessing as mp
import redis
from enum import Enum
import psutil
import aio_msgpack_rpc

from redis_handler import redis_conn
from config import RPC_SERVER, RPC_PORT
from enums import TaskStatus, TaskType, TaskErrorKeyword, WorkerConnectionStatus, TaskInfoKey
from task_utils import update_task_info, remove_used_capacity
from log_handler import logger
from extensions import DetectionTaskManager, TrainingTaskManager

redis_client = redis_conn()


pid_list = list()


def get_task_info(task_id):
    task_info = dict()
    task = redis_client.hget("task_dict", task_id)
    if task:
        task_info = json.loads(task)
    return task_info


def kill_processes(pids):
    for pid in pids:
        proc = psutil.Process(pid)
        for p_child in proc.children(recursive=True):
            p_child.kill()

        proc.kill()


def _monitor_detect_sub_process(pipe, node, task_id):
    try:
        while True:
            if pipe is None:
                break

            data = pipe.stdout.readline().strip()
            if data and len(data):
                if any(keyword.value in data for keyword in TaskErrorKeyword):
                    update_task_info(task_id, TaskInfoKey.LOG.value, f"{task_id}: [Error] : {data}")
                    update_task_info(task_id, TaskInfoKey.STATUS.value, TaskStatus.FAILED.value)
                    remove_used_capacity(node, TaskType.DETECT.value)
                    raise Exception(data)

                else:
                    if not get_task_info(task_id).get("node"):
                        update_task_info(task_id, TaskInfoKey.NODE.value, node)

                    logger.info(f"Task ID: {task_id} Detection Task Information: {data}")

    except Exception as e:
        logger.error(f"Subprocess Error:{e}")
        # raise Exception(e)


def _monitor_training_sub_process(pipe, node, task_id):
    try:
        while True:
            if pipe is None or pipe.poll() is not None:
                break

            data = pipe.stdout.readline()
            data = data.strip()

            if data and len(data):
                if any(keyword.value in data for keyword in TaskErrorKeyword):
                    update_task_info(task_id, TaskInfoKey.LOG.value, f"{task_id} [Error] : {data}")
                    update_task_info(task_id, TaskInfoKey.STATUS.value, TaskStatus.FAILED.value)
                    remove_used_capacity(node, TaskType.TRAINING.value)
                    raise Exception(data)

            else:
                if not get_task_info(task_id).get("node"):
                    update_task_info(task_id, TaskInfoKey.NODE.value, node)

                logger.info(f"Task ID: {task_id} Training Task Information: {data} {len(data)}")

            sleep(1)
    except Exception as e:
        logger.error(f"Subprocess Error:{e}")


class MyServicer:
    async def start_worker(self, task_id, node, task_type, task_info):
        # task_info = {"command": "python detect.py", "task_id": "123"}
        task_command = task_info.get("command")
        if isinstance(task_id, bytes):
            task_id = task_id.decode()

        update_task_info(task_id, TaskInfoKey.NODE.value, node)
        update_task_info(task_id, TaskInfoKey.STATUS.value, TaskStatus.IN_PROGRESS.value)
        update_task_info(task_id, TaskInfoKey.LAST_UPDATED_TIME.value, datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
        update_task_info(task_id, TaskInfoKey.WORK_CONN_STATUS.value, WorkerConnectionStatus.CONNECTING.value)

        if task_type == TaskType.DETECT.value:
            cmd = task_command.split()
            pipe = sp.Popen(cmd, stdout=sp.PIPE, stderr=sp.STDOUT, universal_newlines=True, bufsize=1)

            detect_monitor_thread = threading.Thread(target=_monitor_detect_sub_process, args=(pipe, node, task_id))
            detect_monitor_thread.start()

            # pipe = sp.Popen(cmd, stderr=sp.STDOUT, universal_newlines=True)
            pid = pipe.pid
            update_task_info(task_id, TaskInfoKey.TASK_PID.value, pid)

            msg = {"data": pid, "error_code": "1000"}
            return msg

        elif task_type == TaskType.TRAINING.value:
            cmd = task_command.split()
            pipe = sp.Popen(cmd, stdout=sp.PIPE, stderr=sp.STDOUT, universal_newlines=True)

            training_monitor_thread = threading.Thread(target=_monitor_training_sub_process, args=(pipe, node, task_id))
            training_monitor_thread.start()

            pid = pipe.pid
            update_task_info(task_id, TaskInfoKey.TASK_PID.value, pid)

            msg = {"data": pid, "error_code": "1000"}
            return msg

        else:
            msg = {"error_msg": "Please check the code type.", "error_code": "123"}
            return msg

    async def stop_worker(self, task_id, proc_pid, node, task_type):
        task_info = get_task_info(task_id)

        if task_type == TaskType.DETECT.value:
            task_obj = DetectionTaskManager()
        elif task_type == TaskType.TRAINING.value:
            task_obj = TrainingTaskManager()

        else:
            msg = {"error_msg": "Please check the code type.", "error_code": "123"}
            return msg

        if not psutil.pid_exists(proc_pid):
            if task_info.get("status") != TaskStatus.FAILED.value or task_info.get("status") != TaskStatus.COMPLETED.value:
                update_task_info(task_id, TaskInfoKey.LOG.value, "Task has unexpectedly terminated")
                update_task_info(task_id, TaskInfoKey.STATUS.value, TaskStatus.TERMINATED.value)

        try:
            os.killpg(proc_pid, 1)

        except Exception as e:
            logger.warning(f"kill subprocess for Task ID {task_id}, PID {proc_pid}: {e}")
            kill_processes([proc_pid])

        update_task_info(task_id, TaskInfoKey.LOG.value, f"{task_id}: Task has been canceled")
        update_task_info(task_id, TaskInfoKey.STATUS.value, TaskStatus.CANCELLED.value)
        remove_used_capacity(node, task_type)
        task_obj.update_task(task_id, "task_status", TaskStatus.CANCELLED.value)
        task_obj.close()
        msg = {"error_msg": "success.", "error_code": "1000"}
        return msg


async def rpc_main():
    try:
        server = await asyncio.start_server(aio_msgpack_rpc.Server(MyServicer()), host=RPC_SERVER, port=RPC_PORT)
        idx = 0
        while True:
            if idx == 0:
                logger.warning("Worker Node Start Listening ...")
            await asyncio.sleep(0.1)
            idx += 1
    finally:
        server.close()


if __name__ == '__main__':
    try:
        asyncio.get_event_loop().run_until_complete(rpc_main())

    except KeyboardInterrupt as e:
        logger.warning(f"RPC Server: Keyboard Interrupt {e}")

