import cv2
import numpy as np

class Tracking():
    def __init__(self) -> None:
        super().__init__()
        self.trackers = cv2.MultiTracker()
        self.tracker = cv2.TrackerKCF_create()
