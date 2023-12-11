import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
from PIL import Image
from tempfile import TemporaryDirectory

cudnn.benchmark = True
plt.ion()  # interactive mode
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath("."))))
sys.path.append(parent_dir)

from task_utils import update_task_info, get_task_info
from log_handler import logger
from extensions import TrainingTaskManager
from enums import TaskInfoKey, TaskStatus

data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}


def imshow(inp, title=None):
    """Display image for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated


# def visualize_model_predictions(model, img_path):
#     was_training = model.training
#     model.eval()
#
#     img = Image.open(img_path)
#     img = data_transforms['val'](img)
#     img = img.unsqueeze(0)
#     img = img.to(device)
#
#     with torch.no_grad():
#         outputs = model(img)
#         _, preds = torch.max(outputs, 1)
#         ax = plt.subplot(2, 2, 1)
#         ax.axis('off')
#         ax.set_title(f'Predicted: {class_names[preds[0]]}')
#         imshow(img.cpu().data[0])
#
#         model.train(mode=was_training)


def train_model(task_id, model, criterion, optimizer, scheduler, batch_size, num_epochs=25, dataset_name=None):
    since = time.time()

    result_path = os.path.join("output", "train", task_id)
    if not os.path.exists(result_path):
        os.makedirs(result_path)

    pretrained_model_path = os.path.join('models', task_id)
    if not os.path.exists(pretrained_model_path):
        os.makedirs(pretrained_model_path)

    if not dataset_name:
        logger.info(f"Dataset name is not provided, use default model")
        update_task_info(task_id, TaskInfoKey.LOG.value, f"Dataset name is not provided, use default model")
        dataset_name = "mini"

    if dataset_name == "coco":
        dataset_name = "mini"

    dataset_path = os.path.join('moudules/Resnet/dataset', dataset_name)

    image_datasets = {x: datasets.ImageFolder(os.path.join(dataset_path, x),
                                              data_transforms[x]) for x in ['train', 'val']}

    class_names = image_datasets['train'].classes

    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size,
                                                  shuffle=True, num_workers=4)
                   for x in ['train', 'val']}

    # Create a temporary directory to save training checkpoints
    best_model_params_path = os.path.join(pretrained_model_path, 'best_model_params.pt')
    torch.save(model.state_dict(), best_model_params_path)

    best_acc = 0.0

    train_info_list = dict()
    for epoch in range(num_epochs):

        print(f'Task ID: {task_id} Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)
        logger.info(f"Epoch: {epoch + 1}/{num_epochs}")
        update_task_info(task_id, TaskInfoKey.CURRENT_EPOCH.value, epoch + 1)

        # Each epoch has a training and validation phase

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for i, (inputs, labels) in enumerate(dataloaders[phase]):
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

                # Todo: 加 redis 日志

                logger.warning(
                    f'Epoch: {epoch} Phase: {phase} Iter: {i + 1}/{len(dataloaders[phase])} Loss: {running_loss / dataset_sizes[phase]:.4f} Acc: {running_corrects.double() / dataset_sizes[phase]:.4f}')
                update_task_info(task_id, TaskInfoKey.LOG.value,
                                 f'Epoch: {epoch} Phase: {phase} Iter: {i + 1}/{len(dataloaders[phase])} Loss: {running_loss / dataset_sizes[phase]:.4f} Acc: {running_corrects.double() / dataset_sizes[phase]:.4f}')

                if not f'epoch_{epoch}' in train_info_list:
                    train_info_list[f'epoch_{epoch}'] = list()

                train_info_list[f'epoch_{epoch}'].append({"Loss": str(running_loss / dataset_sizes[phase]),
                                                          "Acc": str(running_corrects.double() / dataset_sizes[phase])})

            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                res_model_path = os.path.join(pretrained_model_path, f'resnet_epoch_{epoch}.pt')
                torch.save(model.state_dict(), res_model_path)
        #                 torch.save(model.state_dict(), best_model_params_path)
        update_task_info(task_id, TaskInfoKey.CURRENT_EPOCH.value, epoch + 1)

    time_elapsed = time.time() - since
    update_task_info(task_id, TaskInfoKey.LOG.value,
                     f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    update_task_info(task_id, TaskInfoKey.LOG.value, f'Best val Acc: {best_acc:4f}')
    logger.info(f'Task ID: {task_id} Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    logger.info(f'Task ID: {task_id} Best val Acc: {best_acc:4f}')

    # load best model weights
    model.load_state_dict(torch.load(best_model_params_path))

    task_obj = TrainingTaskManager()
    task_obj.update_task(task_id, TaskInfoKey.RESULT.value, train_info_list)
    task_obj.update_task(task_id, "task_status", TaskStatus.COMPLETED.value)
    task_obj.close()

    return model


def start(task_id, epoch, batch_size, learning_rate, dataset_name, node, optimizer="Adam", model_name=None):
    # data_dir = 'dataset'

    if not model_name:
        print(f"Model name is not provided, use default model")
        model_path = "pretrain_models/resnet_epoch_3.pt"
    else:
        model_path = os.path.join("models", model_name)
        if not os.path.exists(model_path):
            print(f"{model_name} does not exist, use default model")
            model_path = "pretrain_models/resnet_epoch_3.pt"

    #     device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # 定义本地预训练模型的路径
    model_path = 'moudules/Resnet/models/resnet50.pth'

    # 实例化一个 ResNet-50 模型
    model_ft = models.resnet50()

    # 加载本地预训练模型的权重
    checkpoint = torch.load(model_path)
    model_ft.load_state_dict(checkpoint)

    num_ftrs = model_ft.fc.in_features
    # Here the size of each output sample is set to 2.
    # Alternatively, it can be generalized to ``nn.Linear(num_ftrs, len(class_names))``.
    model_ft.fc = nn.Linear(num_ftrs, 2)

    model_ft = model_ft.to(device)

    criterion = nn.CrossEntropyLoss()

    # Observe that all parameters are being optimized
    if optimizer == "Adam":
        optimizer_ft = optim.Adam(model_ft.parameters(), lr=learning_rate)
    else:
        optimizer_ft = optim.SGD(model_ft.parameters(), lr=learning_rate, momentum=0.9)

    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

    logger.info(f"Resnet model loaded")
    update_task_info(task_id, TaskInfoKey.LOG.value, "Resnet model loaded")

    model = train_model(task_id, model_ft, criterion, optimizer_ft, exp_lr_scheduler, batch_size=batch_size,
                        num_epochs=epoch, dataset_name=dataset_name)

    update_task_info(task_id, TaskInfoKey.LOG.value, f"Training Task: [{task_id}] Already Completed!")
    update_task_info(task_id, TaskInfoKey.STATUS.value, TaskStatus.COMPLETED.value)

    return True


if __name__ == "__main__":
    task_id = "123"
    epoch = 2
    batch_size = 2
    learning_rate = 0.0001
    dataset_name = "mini"
    node = "worker1"
    optimizer = "Adam"

    start(task_id, epoch, batch_size, learning_rate, dataset_name, node)
    # start(batch_size, epoch, lr, optimizer)