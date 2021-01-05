import threading
from face_recognition_sdk.utils.database import FaceRecognitionSystem

detect_lock = threading.Lock()
recog_lock = threading.Lock()

system = FaceRecognitionSystem()