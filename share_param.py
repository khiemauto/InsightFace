import threading
from face_recognition_sdk.utils.database import FaceRecognitionSystem
import logging

logging.basicConfig(filename="insightface.log", level=logging.DEBUG)

detect_lock = threading.Lock()
recog_lock = threading.Lock()

system = FaceRecognitionSystem()