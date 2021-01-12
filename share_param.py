import threading
import logging
import queue
from core import support

# logging.basicConfig(filename="insightface.log", level=logging.DEBUG)

detect_lock = threading.Lock()
recog_lock = threading.Lock()

devconfig = support.get_dev_config()
system = None   #FaceRecognitionSystem

cam_infos = {}
face_infos = {}

bRunning = False

batch_size = 1
stream_queue = queue.Queue(maxsize=15*batch_size)
object_queue = queue.Queue(maxsize=15*batch_size)

GET_FACE_INFO_URL = 'get_face_info'
GET_FACE_INFO_FILE = 'face_info.json'

GET_LIST_DEVICE_URL = 'get_list_device'
GET_LIST_DEVICE_FILE = 'list_device.json'