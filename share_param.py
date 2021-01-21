import threading
import logging
import queue
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
MAX_BATCH_SIZE = 32

STREAM_SIZE = 5
DETECT_SIZE = 5
RECOGN_SIZE = 10
IMSHOW_SIZE = 5

stream_queue = None    #[deviceId, rgb]
detect_queue = None    #[deviceId, bboxs, landmarks, faceCropExpands, rgb]
recogn_queue = None    #{'EventId','UserName','DeviceId,'FaceId','RecordTime','FaceImg'}

push_detect_queue = None    #[deviceId, bboxs, landmarks, faceCropExpands, rgb]

imshow_queue = None    #[title, image]

GET_FACE_INFO_URL = 'get_face_info'
GET_FACE_INFO_FILE = 'face_info.json'

GET_LIST_DEVICE_URL = 'get_list_device'
GET_LIST_DEVICE_FILE = 'list_device.json'

RECOGN_NONE = 0
RECOGN_LOCAL = 1
RECOGN_CLOUD = 2

#Hyperbol blur
qi = 1345.33325
b = 0.52109685
di = 2.3316e-04