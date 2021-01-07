import threading
import logging
import queue
from minio import Minio
from core import support

client_minio: Minio = None
logging.basicConfig(filename="insightface.log", level=logging.DEBUG)

detect_lock = threading.Lock()
recog_lock = threading.Lock()

devconfig = support.get_dev_config()
system = None   #FaceRecognitionSystem

cam_infos = {}
face_infos = {}

bRunning = True

batch_size = 1
stream_queue = queue.Queue(maxsize=15*batch_size)
object_queue = queue.Queue(maxsize=15*batch_size)

