import queue
import cv2
import argparse
import time
import datetime
import threading
import os
import sys
from typing import Tuple
from face_recognition_sdk.utils.database import FaceRecognitionSystem
import uvicorn
import socketserver
import http.server
from api.rest import FaceRecogAPI
import share_param
from core import support, pushserver


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
        while share_param.stream_queue.qsize() > share_param.STREAM_SIZE*share_param.batch_size:
            share_param.stream_queue.get()
        share_param.stream_queue.put(
            [deviceId, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)])

    if cap:
        cap.release()


def detect_thread_fun():
    while share_param.bRunning:
        # print( 'line', inspect.getframeinfo(inspect.currentframe()).lineno)
        time.sleep(0.001)
        totalTime = time.time()
        if share_param.stream_queue.qsize() < share_param.batch_size:
            continue

        detect_inputs = []  # [[deviceId, rbg]]
        preTime = time.time()

        for batchId in range(share_param.batch_size):
            deviceId, rgb = share_param.stream_queue.get()
            # xstart = (bgr.shape[1] - bgr.shape[0])//2
            # bgr = bgr[:, xstart: xstart + bgr.shape[0]]
            # print(rgb.shape)
            # rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            detect_inputs.append([deviceId, rgb])

        preTime = time.time()
        for batchId, (deviceId, rgb) in enumerate(detect_inputs):
            share_param.detect_lock.acquire()
            bboxes, landmarks = share_param.system.sdk.detect_faces(rgb)
            share_param.detect_lock.release()
            bbox_keeps = []
            landmark_keeps = []
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
                bbox_keeps.append(bbox)
                landmark_keeps.append(landmark)

            while share_param.detect_queue.qsize() > share_param.DETECT_SIZE*share_param.batch_size:
                share_param.detect_queue.get()
            share_param.detect_queue.put(
                [deviceId, rgb, bbox_keeps, landmark_keeps])

        print("Detect Time:", time.time() - totalTime)


def recogn_thread_fun():
    while share_param.bRunning:
        time.sleep(0.001)
        totalTime = time.time()
        if share_param.detect_queue.qsize() < share_param.batch_size:
            continue

        recogn_inputs = []  # [[deviceId, bgr, rgb, bboxs, landmarks]]
        for i in range(share_param.batch_size):
            recogn_inputs.append(share_param.detect_queue.get())

        for batchId, (deviceId, rgb, bboxs, landmarks) in enumerate(recogn_inputs):
            names = []
            similarities = []
            faceCrops = []
            faceCropExpands = []

            for bbox, landmark in zip(bboxs, landmarks):
                if not share_param.system.photoid_to_username_photopath:
                    names.append('unknown')
                    similarities.append(0.0)
                    continue

                face = share_param.system.sdk.align_face(rgb, landmark)
                share_param.recog_lock.acquire()
                descriptor = share_param.system.sdk.get_descriptor(face)
                indicies, distances = share_param.system.sdk.find_most_similar(
                    descriptor)
                share_param.recog_lock.release()
                user_name = share_param.system.get_user_name(indicies[0])
                names.append(user_name)
                similarities.append(distances[0])
                faceCrops.append(rgb[int(bbox[1]):int(
                    bbox[3]), int(bbox[0]):int(bbox[2])])
                faceW = abs(bbox[2] - bbox[0])
                faceH = abs(bbox[3] - bbox[1])
                expandLeft = max(0, bbox[0] - faceW/3)
                expandTop = max(0, bbox[1] - faceH/3)
                expandRight = min(bbox[2] + faceW/3, rgb.shape[1])
                expandBottom = min(bbox[3] + faceH/3, rgb.shape[0])
                faceCropExpands.append(rgb[int(expandTop):int(
                    expandBottom), int(expandLeft):int(expandRight)])

            for (box, staffId, score, faceCrop, faceCropExpand) in zip(bboxs, names, similarities, faceCrops, faceCropExpands):
                if score > share_param.devconfig["DEV"]["face_reg_score"]:
                    cv2.rectangle(rgb, (int(box[0]), int(box[1])), (int(
                        box[2]), int(box[3])), (0, 255, 0), 2)
                    y = box[1] - 15 if box[1] - 15 > 15 else box[1] + 15
                    cv2.putText(rgb, "{} {:03.3f}".format(staffId, score), (int(box[0]), int(y)), cv2.FONT_HERSHEY_SIMPLEX,
                                0.75, (0, 255, 0), 2)
                else:
                    staffId = "unknown"
                    cv2.rectangle(rgb, (int(box[0]), int(
                        box[1])), (int(box[2]), int(box[3])), (0, 0, 255), 2)
                    y = box[1] - 15 if box[1] - 15 > 15 else box[1] + 15
                    cv2.putText(rgb, "{} {:03.3f}".format(staffId, score), (int(box[0]), int(y)), cv2.FONT_HERSHEY_SIMPLEX,
                                0.75, (0, 0, 255), 2)

                pushserver.add_object_queue(staffId, deviceId, datetime.datetime.now(
                ).strftime("%Y-%m-%d %H:%M:%S"), faceCropExpand)

            support.custom_imshow(str(deviceId), cv2.resize(rgb, (500, 500)))

            print("Recogn Time:", time.time() - totalTime)


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
    share_param.batch_size = len(share_param.cam_infos)
    share_param.stream_queue = queue.Queue(
        maxsize=share_param.STREAM_SIZE*share_param.batch_size+3)
    share_param.detect_queue = queue.Queue(
        maxsize=share_param.DETECT_SIZE*share_param.batch_size+3)
    share_param.recogn_queue = queue.Queue(
        maxsize=share_param.RECOGN_SIZE*share_param.batch_size+3)

    stream_threads = []
    for deviceID, camURL in share_param.cam_infos.items():
        stream_threads.append(threading.Thread(
            target=stream_thread_fun, daemon=True, args=(deviceID, camURL)))

    detect_thread = threading.Thread(
        target=detect_thread_fun, daemon=True, args=())
    recogn_thread = threading.Thread(
        target=recogn_thread_fun, daemon=True, args=())
    pushserver_thread = threading.Thread(
        target=pushserver.pushserver_thread_fun, daemon=True, args=())
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

    share_param.app = FaceRecogAPI(
        share_param.system, folders_path, db_folder_path)
    uvicorn.run(share_param.app, host=share_param.devconfig["APISERVER"]
                ["host"], port=share_param.devconfig["APISERVER"]["port"])
    share_param.bRunning = False
    fileserver.shutdown()
    cv2.destroyAllWindows()
