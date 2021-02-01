import cv2
import numpy as np
from typing import List, Dict, Tuple

class Tracking():
    def __init__(self) -> None:
        super().__init__()
        self.trackers = {}  #trackid: [Tracker]
        self.threshiou = 0.5
        self.maxid = 1
    
    def newsession(self, frame, detectboxs: List[int]):
        """
        box : x,y,w,h
        """

        trackboxs = {}

        for trackid in list(self.trackers):
            (success, trackbox) = self.trackers[trackid].update(frame)
            # if success:
            trackboxs[trackid] = trackbox
            # else:
            #     del self.trackers[trackid]

        track_detect_iou = {}   #(trackidmaxiou: detectboxid, detectbox, maxiou)
        detect_track_iou = {}   #(detectboxid: trackidmaxiou)

        for detectboxid, detectbox in enumerate(detectboxs):
            maxiou = 0
            trackidmaxiou = -1
            for trackid, trackbox in trackboxs.items():
                # print(detectbox, trackbox)
                xyxydetectbox = (detectbox[0], detectbox[1], detectbox[0]+detectbox[2], detectbox[1]+ detectbox[3])
                xyxytrackbox = (trackbox[0], trackbox[1], trackbox[0]+trackbox[2], trackbox[1]+ trackbox[3])
                iou = self.bb_intersection_over_union(xyxydetectbox, xyxytrackbox)
                if iou > maxiou:
                    maxiou = iou
                    trackidmaxiou = trackid
                    
            print(maxiou)
            if maxiou<self.threshiou:
                continue

            if trackidmaxiou not in track_detect_iou:
                track_detect_iou[trackidmaxiou] = (detectboxid, detectbox, maxiou)
                detect_track_iou[detectboxid] = trackidmaxiou
            else:
                if maxiou > track_detect_iou[trackidmaxiou][2]:
                    track_detect_iou[trackidmaxiou] = (detectboxid, detectbox, maxiou)
                    detect_track_iou[detectboxid] = trackidmaxiou
        
        trackid_bboxes = []
        #Clean tracker
        for trackid in list(self.trackers):
            if trackid in track_detect_iou:
                self.trackers[trackid] = cv2.legacy.TrackerMOSSE_create()
                self.trackers[trackid].init(frame, track_detect_iou[trackid][1])
                # self.trackers[trackid][1] = 0
                trackid_bboxes.append((trackid, track_detect_iou[trackid][1]))
            else:
                del self.trackers[trackid]
            
        #Create track with new detect
        for detectboxid, detectbox in enumerate(detectboxs):
            if detectboxid not in detect_track_iou:
                self.maxid += 1
                self.trackers[self.maxid] = cv2.legacy.TrackerMOSSE_create()
                self.trackers[self.maxid].init(frame, detectbox)
                trackid_bboxes.append((self.maxid, detectbox))

        # print(len(trackid_bboxes))
        return trackid_bboxes

    def release(self):
        self.trackers.clear()

    def update(self, frame):
        ret = []
        for trackid in list(self.trackers):
            (success, boxes) = self.trackers[trackid].update(frame)
            # if success:
            ret.append((trackid, boxes))
            # else:
                # del self.trackers[trackid]
        return ret
        
    def bb_intersection_over_union(self, boxA, boxB):
        # determine the (x, y)-coordinates of the intersection rectangle
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])

        # compute the area of intersection rectangle
        interArea = abs(max((xB - xA, 0)) * max((yB - yA), 0))
        if interArea == 0:
            return 0
        # compute the area of both the prediction and ground-truth
        # rectangles
        boxAArea = abs((boxA[2] - boxA[0]) * (boxA[3] - boxA[1]))
        boxBArea = abs((boxB[2] - boxB[0]) * (boxB[3] - boxB[1]))

        # compute the intersection over union by taking the intersection
        # area and dividing it by the sum of prediction + ground-truth
        # areas - the interesection area
        iou = interArea / float(boxAArea + boxBArea - interArea)

        # return the intersection over union value
        return iou
