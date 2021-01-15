import threading
import logging
import queue
from dataclasses import dataclass
from typing import List
import numpy as np

# logging.basicConfig(filename="insightface.log", level=logging.DEBUG)

detect_lock = threading.Lock()
recog_lock = threading.Lock()

devconfig = None
system = None   #FaceRecognitionSystem
app = None  #FaceRecogAPI

cam_infos = {}
face_infos = {}

bRunning = False

batch_size = 1

STREAM_SIZE = 5
DETECT_SIZE = 5
RECOGN_SIZE = 10

stream_queue = None    #[deviceId, rgb]
detect_queue = None    #[deviceId, rgb, bboxs, landmarks]
recogn_queue = None    #{'EventId','UserName','DeviceId,'FaceId','RecordTime','FaceImg'}

GET_FACE_INFO_URL = 'get_face_info'
GET_FACE_INFO_FILE = 'face_info.json'

GET_LIST_DEVICE_URL = 'get_list_device'
GET_LIST_DEVICE_FILE = 'list_device.json'

#Hyperbol blur
qi = 1345.33325
b = 0.52109685
di = 2.3316e-04

#Meta data
# @dataclass
# class FaceData:
#     def __init__(self) -> None:
#         self.FaceID: int
#         self.UserName: str
#         self.bbox: np.ndarray
#         self.landmark: np.ndarray
#         self.reg_score: float
#         self.FaceAlign: np.ndarray
#         self.FaceExpand: np.ndarray

# # @dataclass
# class FrameData:
#     def __init__(self) -> None:
#         self.Frame: np.ndarray
#         self.DeviceID: int = -1
#         self.FrameID: int = -1
#         self.FaceDataList: List[FaceData] = []

    
