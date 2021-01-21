"""
Author: Khiem Tran,
Email: khiembka1992@gmail.com",
Date: 2020/01/15
"""
import queue
import cv2
import argparse
import time
import datetime
import threading
import os
import sys
from typing import Tuple

from fastapi import params
from face_recognition_sdk.utils.database import FaceRecognitionSystem
import uvicorn
import socketserver
import http.server
from api.rest import FaceRecogAPI
import share_param
from core import support, pushserver
import numpy as np
import requests
import json
from core.tracking import Tracking

test = Tracking()

ap = argparse.ArgumentParser()
ap.add_argument("-fp", "--folders_path", default=None,
                help="path to save folders with images")
ap.add_argument("-dbp", "--db_folder_path", default="database",
                help="path to save database")
ap.add_argument("-rdb", "--reload_db", type=int, default=0,
                help="reload database")
args = vars(ap.parse_args())


def initiation() -> Tuple[dict, dict]:
    if not share_param.devconfig["DEV"]["print"]:
        sys.stdout = open(os.devnull, 'w')

    cam_dicts = support.get_cameras()
    face_dicts = support.get_faces()

    return cam_dicts, face_dicts


def stream_thread_fun(deviceID: int, camURL: str):
    deviceId = deviceID
    cap = cv2.VideoCapture(camURL, cv2.CAP_FFMPEG)

    if not cap or not cap.isOpened():
        print(f"[ERROR] Camera not open {camURL}")
    
    else:
        print(f"[INFO] Camera opened {camURL}")

    FrameID = 1
    timeStep = 1/15  # 10FPS
    lastFrame = time.time()
    lastGood = time.time()

    while share_param.bRunning:
        time.sleep(0.01)
        if time.time() - lastGood > 300:
            print("[INFO] Restart cam:", camURL)
            cap.open(camURL)
            lastGood = time.time()

        if cap is None or not cap.isOpened():
            continue

        FrameID += 1
        if time.time() - lastFrame > timeStep:
            lastFrame = time.time()
            (grabbed, frame) = cap.read()
            if not grabbed or frame is None or frame.size == 0:
                continue
        else:
            cap.grab()
            continue

        lastGood = time.time()
        xstart = (frame.shape[1] - frame.shape[0])//2
        frame = frame[:, xstart: xstart + frame.shape[0]]
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        while share_param.stream_queue.qsize() > share_param.STREAM_SIZE*share_param.batch_size:
            share_param.stream_queue.get()
        share_param.stream_queue.put((deviceId, frame))

    if cap:
        cap.release()


def detect_thread_fun():
    if share_param.devconfig["DEV"]["option_detection"] != 1:
        return
    totalTime = time.time()
    while share_param.bRunning: 
        # print( 'line', inspect.getframeinfo(inspect.currentframe()).lineno)
        totalTime = time.time()
        time.sleep(0.001)
        if share_param.stream_queue.qsize() < share_param.batch_size:
            continue

        detect_inputs = []  # [[deviceId, rbg]]
        preTime = time.time()

        for batchId in range(share_param.batch_size):
            detect_input = share_param.stream_queue.get()
            detect_inputs.append(detect_input)

        preTime = time.time()
        rgbs = []
        for (deviceId, rgb) in detect_inputs:
            rgbs.append(rgb)

        share_param.detect_lock.acquire()
        bboxes_batch, landmarks_batch = share_param.system.sdk.detect_faces_batch(rgbs)
        share_param.detect_lock.release()

        del rgbs

        for bboxes, landmarks, (deviceId, rgb) in zip(bboxes_batch, landmarks_batch, detect_inputs):
            bbox_keeps = []
            landmark_keeps = []
            faceCropExpand_keeps = []

            draw_bboxs = []
            draw_landmarks = []

            for bbox, landmark in zip(bboxes, landmarks):
                # Skip blur face
                imgcrop = rgb[int(bbox[1]):int(bbox[3]),
                              int(bbox[0]):int(bbox[2])]
                if imgcrop is None or imgcrop.size == 0:
                    continue

                faceW = abs(bbox[2] - bbox[0])
                faceH = abs(bbox[3] - bbox[1])

                notblur = cv2.Laplacian(imgcrop, cv2.CV_32F).var()
                threshblur = support.get_blur_var(faceW*faceH)
                if notblur < 0.8*threshblur:
                    continue

                expandLeft = max(0, bbox[0] - faceW/3)
                expandTop = max(0, bbox[1] - faceH/3)
                expandRight = min(bbox[2] + faceW/3, rgb.shape[1])
                expandBottom = min(bbox[3] + faceH/3, rgb.shape[0])
                faceCropExpand = rgb[int(expandTop):int(expandBottom), int(expandLeft):int(expandRight)].copy()
                faceCropExpand_keeps.append(faceCropExpand)

                draw_bboxs.append(bbox)
                draw_landmarks.append(landmark)

                #Mov abs position to faceCropExpand coordinate
                relbox = [bbox[0]-expandLeft, bbox[1]-expandTop, bbox[2]-expandLeft, bbox[3]-expandTop, bbox[4]]
                rellandmark = landmark.reshape((5,2), order= "F")
                rellandmark = rellandmark-[expandLeft, expandTop]
                rellandmark = rellandmark.ravel(order= "F")

                bbox_keeps.append(np.asarray(relbox))
                landmark_keeps.append(np.asarray(rellandmark))
                
            #Draw
            for draw_bbox in draw_bboxs:
                cv2.rectangle(rgb, (int(draw_bbox[0]), int(draw_bbox[1])), (int(draw_bbox[2]), int(draw_bbox[3])), (0, 255, 0), 2)

            while share_param.imshow_queue.qsize() > share_param.IMSHOW_SIZE*share_param.batch_size:
                    share_param.imshow_queue.get()
            share_param.imshow_queue.put((str(deviceId), cv2.resize(rgb, (500, 500))))
            #Skip for emtpy
            if len(bbox_keeps)==0 or len(landmark_keeps)==0:
                continue
            
            if share_param.devconfig["DEV"]["option_recogition"] == share_param.RECOGN_LOCAL:
                while share_param.detect_queue.qsize() > share_param.DETECT_SIZE*share_param.batch_size:
                    share_param.detect_queue.get()
                share_param.detect_queue.put((deviceId, bbox_keeps, landmark_keeps, faceCropExpand_keeps, None))
            
            elif share_param.devconfig["DEV"]["option_recogition"] == share_param.RECOGN_CLOUD:
                while share_param.push_detect_queue.qsize() > share_param.DETECT_SIZE*share_param.batch_size:
                    share_param.push_detect_queue.get()
                share_param.push_detect_queue.put((deviceId, bbox_keeps, landmark_keeps, faceCropExpand_keeps, None))

        print("Detect Time:", time.time() - totalTime)

def recogn_thread_fun():
    if share_param.devconfig["DEV"]["option_recogition"] != 1:
        return
    
    totalTime = time.time()
    while share_param.bRunning:
        totalTime = time.time()
        time.sleep(0.001)
        # print("Recogn Time:", time.time() - totalTime)
        if share_param.detect_queue.qsize() < share_param.batch_size:
            continue

        recogn_inputs = []
        for i in range(share_param.batch_size):
            recogn_inputs.append(share_param.detect_queue.get())

        faceInfos = []
        faceAligns = []

        preTime = time.time()

        for deviceId, bboxs, landmarks, faceCropExpands, rgb in recogn_inputs:
            for bbox, landmark, faceCropExpand in zip(bboxs, landmarks, faceCropExpands):
                if faceCropExpand is None or faceCropExpand.size ==0 or landmark is None or landmark.size==0:
                    continue

                faceAlign = share_param.system.sdk.align_face(faceCropExpand, landmark)

                faceInfos.append([deviceId, bbox, landmark, faceCropExpand])
                faceAligns.append(faceAlign)

        if len(faceAligns) == 0:
            continue

        print("Align Time:", time.time() - preTime)
        preTime = time.time()
        share_param.recog_lock.acquire()
        descriptors = share_param.system.sdk.get_descriptor_batch(faceAligns)
        share_param.recog_lock.release()

        del faceAligns

        print("Description Time:", time.time() - preTime)
        preTime = time.time()
        indicies = []
        distances = []
        if not share_param.system.photoid_to_username_photopath:
            for faceInfo in faceInfos:
                faceInfo.append('unknown')
                faceInfo.append(0.0)
        else:
            indicies, distances = share_param.system.sdk.find_most_similar_batch(descriptors)

            for faceInfo, indicie, distance in zip(faceInfos, indicies, distances):
                user_name = share_param.system.get_user_name(indicie[0])
                faceInfo.append(user_name)
                faceInfo.append(distance[0])

        print("Similar Time:", time.time() - preTime)
        preTime = time.time()

        for deviceId, bbox, landmark, faceCropExpand, user_name, score in faceInfos:
            if score > share_param.devconfig["DEV"]["face_reg_score"]:
                # while share_param.imshow_queue.qsize() > share_param.IMSHOW_SIZE*share_param.batch_size:
                #     share_param.imshow_queue.get()
                # share_param.imshow_queue.put((str(deviceId) + user_name, faceCropExpand))
                print(user_name)
            else:
                user_name = "unknown"
            pushserver.add_recogn_queue(user_name, deviceId, datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), faceCropExpand)
        
        print("Push Time:", time.time() - preTime)
        
        # print("Recogn Time:", time.time() - totalTime)

def imshow_thread_fun():        
    while share_param.bRunning:
        time.sleep(0.01)
        while not share_param.imshow_queue.empty():
            title, image = share_param.imshow_queue.get()
            if share_param.devconfig["DEV"]["imshow"]:
                cv2.imshow(title, image)
            cv2.waitKey(10)


if __name__ == '__main__':
    folders_path = args["folders_path"]
    db_folder_path = args["db_folder_path"]

    share_param.devconfig = support.get_dev_config()
    share_param.system = FaceRecognitionSystem(folders_path)

    if args["reload_db"]:
        share_param.system.create_database_from_folders(folders_path)
        share_param.system.save_database(db_folder_path)
    share_param.system.load_database(db_folder_path)

    share_param.cam_infos, share_param.face_infos = initiation()
    # print(len(share_param.cam_infos))
    share_param.batch_size = len(share_param.cam_infos)
    share_param.stream_queue = queue.Queue(
        maxsize=share_param.STREAM_SIZE*share_param.batch_size+3)
    share_param.detect_queue = queue.Queue(
        maxsize=share_param.DETECT_SIZE*share_param.batch_size+3)
    share_param.recogn_queue = queue.Queue(
        maxsize=share_param.RECOGN_SIZE*share_param.batch_size+3)

    share_param.push_detect_queue = queue.Queue(
        maxsize=share_param.DETECT_SIZE*share_param.batch_size+3)

    share_param.imshow_queue = queue.Queue(
        maxsize=share_param.IMSHOW_SIZE*share_param.batch_size+3)

    stream_threads = []
    for deviceID, camURL in share_param.cam_infos.items():
        # if deviceID == 41:
        stream_threads.append(threading.Thread(
            target=stream_thread_fun, daemon=True, args=(deviceID, camURL)))

    detect_thread = threading.Thread(
        target=detect_thread_fun, daemon=True, args=())
    recogn_thread = threading.Thread(
        target=recogn_thread_fun, daemon=True, args=())
    pushserver_thread = threading.Thread(
        target=pushserver.pushserver_thread_fun, daemon=True, args=())

    imshow_thread = threading.Thread(
        target=imshow_thread_fun, daemon=True, args=())

    fileserver = socketserver.TCPServer(
        (share_param.devconfig["FILESERVER"]["host"], share_param.devconfig["FILESERVER"]["port"]), http.server.SimpleHTTPRequestHandler)
    file_thread = threading.Thread(
        target=fileserver.serve_forever, daemon=True, args=())

    share_param.bRunning = True

    file_thread.start()
    for stream_thread in stream_threads:
        stream_thread.start()
    detect_thread.start()
    recogn_thread.start()
    pushserver_thread.start()
    imshow_thread.start()

    if share_param.devconfig["APISERVER"]["is_server"]:
        share_param.app = FaceRecogAPI(
            share_param.system, folders_path, db_folder_path)
        uvicorn.run(share_param.app, host=share_param.devconfig["APISERVER"]
                    ["host"], port=share_param.devconfig["APISERVER"]["port"])
    else:
        detect_thread.join()
    share_param.bRunning = False
    fileserver.shutdown()
    cv2.destroyAllWindows()
