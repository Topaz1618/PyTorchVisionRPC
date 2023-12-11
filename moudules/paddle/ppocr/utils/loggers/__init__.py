import logging
logging.basicConfig(filename='file_calls.log', level=logging.INFO)
logging.info(f'File {__file__} was called.')
from .vdl_logger import VDLLogger
from .wandb_logger import WandbLogger
from .loggers import Loggers
