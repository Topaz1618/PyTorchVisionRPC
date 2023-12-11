import os
import sys
from time import sleep

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

from enums import TaskStatus, TaskInfoKey
from task_utils import update_task_info, get_task_info
from log_handler import logger


iterations_per_epoch = 10


def demo(task_id, task_info, args):
    update_task_info(task_id, TaskInfoKey.EPOCH.value, args.epoch)
    update_task_info(task_id, TaskInfoKey.ITERATIONS.value, iterations_per_epoch)

    l = []
    for epoch in range(args.epoch):
        logger.info(f"Processing Task: [{task_id}], Epoch: {epoch +1}/{args.epoch}")
        update_task_info(task_id, TaskInfoKey.CURRENT_EPOCH.value, epoch + 1)
        update_task_info(task_id, TaskInfoKey.LOG.value,
                         f"Processing Task: [{task_id}], Epoch: {epoch +1}/{args.epoch}")

        for idx in range(iterations_per_epoch):
            logger.info(f"Processing Task: [{task_id}], Epoch: {epoch + 1}/{args.epoch}  Idx: {idx+1}/{iterations_per_epoch}")
            update_task_info(task_id, TaskInfoKey.CURRENT_ITERATION.value, idx+1)
            update_task_info(task_id, TaskInfoKey.LOG.value,
                             f"Processing Task: [{task_id}], Epoch: {epoch + 1}/{args.epoch}  Idx: {idx+1}/{iterations_per_epoch}")

        sleep(1)

        l.append(epoch)
    update_task_info(task_id, TaskInfoKey.STATUS.value, TaskStatus.COMPLETED.value)
    update_task_info(task_id, TaskInfoKey.LOG.value, f"Training Task: 【{task_id}】 Already Completed!")
    update_task_info(task_id, TaskInfoKey.RESULT.value, l)

    return True


if __name__ == "__main__":
    task_id = "4ba5ff7d-f43c-4676-9dd8-57dc4339f1d6"
    task_info = {"task_id": task_id}
