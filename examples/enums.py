from enum import Enum


class FileType(Enum):
    PDF = 0
    OTHERS = 1


class TaskErrorKeyword(Enum):
    ERROR = "Error"
    ERR = "Err"
    RAISE = "raise"
    ERROR_CASE_INSENSITIVE = "error"
    # TRACEBACK = "Traceback"

class UploadStatus(Enum):
    IN_PROGRESS = 1
    COMPLETED = 2
    FAILED = 3

class ModelSource(Enum):
    BUILT_IN = 1
    CUSTOM = 2


class DetectionAlgorithm(Enum):
    YOLO = "YOLO"
    MaskRCNN = "Mask_RCNN"
    RESNET = "Resnet"
    PADDLEOCR = "PaddleOCR"
    TESSERACT = "Tesseract"


class TrainModelType(Enum):
    YOLO = "YOLO"
    MaskRCNN = "Mask_RCNN"
    RESNET = "Resnet"
    PADDLEOCR = "PaddleOCR"


class MaterialContentType(Enum):
    ID_CARD = "居民身份证"
    LAND_MAP = "宗地图"
    LAND_MAP_AREA = "宗地面积"
    HOUSEHOLD_REGISTER = "户口"
    HOUSE_FLOOR_PLAN = "房屋平面图"
    REAL_ESTATE_APPLICATION = "不动产调查登记申请表"
    UNKNOWN = "Unknown"

    def get_text_format(self):
        return self.name


class MaterialType(Enum):
    ID_CARD = "身份证"
    LAND_MAP = "宗地图"
    HOUSEHOLD_REGISTER = "户口本"
    HOUSE_FLOOR_PLAN = "房屋平面图"
    REAL_ESTATE_APPLICATION = "不动产申请表"
    UNKNOWN = "Unknown"

    def get_text_format(self):
        return self.name


class FileFormatType(Enum):
    DOCX = ".docx"
    PDF = ".pdf"
    JPG = ".jpg"
    PNG = ".png"
    XLSX = ".xlsx"
    CSV = ".csv"
    UNKNOWN = "Unknown"

    def get_text_format(self):
        return self.name


class MaterialTitleType(Enum):
    LAND_MAP = "土地权利人"
    REAL_ESTATE_APPLICATION = "不动产单元号"
    UNKNOWN = "Unknown"

    def get_text_format(self):
        return self.name


class TaskType(Enum):
    DETECT = 1
    TRAINING = 2

    def get_text_format(self):
        return self.name


class TaskManagementType(Enum):
    START = 1
    STOP = 2


class TaskInfoKey(Enum):
    LOG = 'log'
    STATUS = 'status'
    NODE = 'node'
    RESULT = 'res'
    TASK_PID = 'task_pid'

    LAST_UPDATED_TIME = 'last_updated_time'
    WORK_CONN_STATUS = 'work_conn_status'

    TOTAL_FILE_COUNT = 'total_file_count'
    PROCESSED_FILE_COUNT = 'processed_file_count'

    EPOCH = 'epoch'
    ITERATIONS = 'iterations'
    CURRENT_EPOCH = 'current_epoch'
    CURRENT_ITERATION = 'current_iteration'


class TaskStatus(Enum):
    PENDING = 1
    IN_PROGRESS = 2
    COMPLETED = 3
    FAILED = 4
    CANCELLED = 5
    RETRYING = 6
    TIMEOUT = 7
    TERMINATED = 8


class WorkerStatus(Enum):
    IDLE = 1
    BUSY = 2
    OFFLINE = 3
    ERROR = 4
    RETRYING = 5


class WorkerConnectionStatus(Enum):
    CONNECTING = 1
    CONNECTED = 2
    FAILED = 3
    RETRYING = 4


class TaskKeyType(Enum):
    LOG = "log"

class ModelSource(Enum):
    BUILT_IN = 1
    CUSTOM = 2

class DatasetSource(Enum):
    BUILT_IN = 1
    CUSTOM = 2



file_title_keywords = {
    MaterialType.LAND_MAP: ['宗地代码', '土地权利人', '所在图幅号', '宗地面积', 'J1', 'J2', 'J3', 'J4'],
    MaterialType.REAL_ESTATE_APPLICATION:  ['不动产调查登记申请表', '预编号', '宗地代码', '不动产单元号', '权利人', '权利人类型', '证件种类', '证件号', '联系电话', '权利人身份', '法定代表人或负责人姓名', '电话', '代理人姓名', '户口簿', '权利类型', '权属来源证明材料', '土地来源证明材料', '权属证件号', '权属证件发证时间', '权利性质', '共有/共用权利人情况', '批准用途', '实际用途', '地类编码', '批准面积', '宗地面积', '房屋竣工时间', '房屋性质', '房屋状况', '幢号', '总层数', '所在层', '房屋结构', '占地面积', '建筑面积', '专有建筑面积', '分摊建筑面积', '权属来源', '墙体归属', '东', '南', '西', '北']
}
