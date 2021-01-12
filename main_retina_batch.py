import queue
import cv2
import argparse
import pickle
import time
import datetime
import math
import threading
import requests
import json
import torch
import os
import sys
from typing import Tuple
import numpy as np
from core import support
from PIL import Image
from face_recognition_sdk.utils.database import FaceRecognitionSystem
import core
import uvicorn
import inspect
import socketserver, http.server
from api.rest import FaceRecogAPI
import share_param
import io

GET_FACE_INFO_URL = 'get_face_info'
GET_FACE_INFO_FILE = 'face_info.json'

GET_LIST_DEVICE_URL = 'get_list_device'
GET_LIST_DEVICE_FILE = 'list_device.json'

ap = argparse.ArgumentParser()
ap.add_argument("-fp", "--folders_path", default=None,
                help="path to save folders with images")
ap.add_argument("-dbp", "--db_folder_path", default="database",
                help="path to save database")
ap.add_argument("-rdb", "--reload_db", type=int, default=0,
                help="reload database")
args = vars(ap.parse_args())
folders_path = args["folders_path"]
db_folder_path = args["db_folder_path"]

share_param.system = FaceRecognitionSystem(folders_path)

# create, save and load database initialized from folders containing user photos
if args["reload_db"]:
    share_param.system.create_database_from_folders(folders_path)
    share_param.system.save_database(db_folder_path)
share_param.system.load_database(db_folder_path)

app = FaceRecogAPI(share_param.system, folders_path, db_folder_path)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('Running on device: {}'.format(device))


def custom_imshow(title: str, image: np.ndarray):
    if support.get_dev_config()["DEV"]["imshow"]:
        cv2.imshow(title, image)
        cv2.waitKey(1)


def initiation() -> Tuple[dict, dict]:
    if not support.get_dev_config()["DEV"]["print"]:
        sys.stdout = open(os.devnull, 'w')

    camera_datas = support.get_infor(GET_LIST_DEVICE_URL, GET_LIST_DEVICE_FILE)
    face_datas = support.get_infor(GET_FACE_INFO_URL, GET_FACE_INFO_FILE)
    cam_dicts = {}
    face_dicts = {}
    for camera_data in camera_datas:
        cam_dicts[camera_data['DeviceId']] = camera_data['LinkRTSP']

    for face_data in face_datas:
        face_dicts[face_data["StaffCode"]] = face_data

    return cam_dicts, face_dicts


def stream_thread_fun_oneCam(deviceID: int, camURL: str):
    deviceId = deviceID
    cap = cv2.VideoCapture(camURL, cv2.CAP_FFMPEG)
    if cap is None or not cap.isOpened():
        print("[Error] Can't connect to {}".format(camURL))
        return

    FrameID = 1
    timeStep = 1/10 #10FPS
    preStep = time.time()

    while share_param.bRunning:
        time.sleep(0.01)
        FrameID += 1
        if time.time() - preStep > timeStep:
            preStep = time.time()
            (grabbed, frame) = cap.read()
            if not grabbed or frame is None or frame.size == 0:
                continue
        else:
            cap.grab()
            continue

        while share_param.stream_queue.qsize() > 5*share_param.batch_size:
            share_param.stream_queue.get()
        share_param.stream_queue.put([deviceId, frame])
    cap.release()


def tracking_thread_fun():
    small_scale = 1
    while share_param.bRunning:
        # print( 'line', inspect.getframeinfo(inspect.currentframe()).lineno)
        time.sleep(0.001)
        totalTime = time.time()
        if share_param.stream_queue.qsize() < share_param.batch_size:
            continue
        frameList = []  # [DeviceID, Image]
        rgbList = []  # [DeviceID, Image]
        small_rgbList = []  # [DeviceID, Image]
        preTime = time.time()

        deviceIdList = []
        for i in range(share_param.batch_size):
            deviceId, frame = share_param.stream_queue.get()
            xstart = (frame.shape[1] - frame.shape[0])//2
            frame = frame[:, xstart: xstart + frame.shape[0]]

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            small_rgb = cv2.resize(rgb, (0, 0), fx=small_scale, fy=small_scale)
            frameList.append(frame)
            rgbList.append(rgb)
            small_rgbList.append(small_rgb)
            deviceIdList.append(deviceId)

        boxDeviceID = []
        boxframeID = []
        boxes = []
        points = []
        scores = []
        faceIds = []
        staffIds = []
        faceCropList = []
        faceCropExpandList = []

        preTime = time.time()
        for frameID, (image, deviceId) in enumerate(zip(small_rgbList, deviceIdList)):
            share_param.detect_lock.acquire()
            bboxes, landmarks = share_param.system.sdk.detect_faces(image)
            share_param.detect_lock.release()
            bbox_keeps = []
            landmark_keeps = []
            for bbox, landmark in zip(bboxes, landmarks):
                # Skip blur face
                imgcrop = image[int(bbox[1]):int(bbox[3]),
                                int(bbox[0]):int(bbox[2])]
                if imgcrop is None or imgcrop.size == 0:
                    continue
                notblur = cv2.Laplacian(imgcrop, cv2.CV_32F).var()

                # Hyperbol blur
                faceW = abs(bbox[2] - bbox[0])
                faceH = abs(bbox[3] - bbox[1])
                # print(faceW*faceH, notblur)
                t = faceW*faceH
                qi = 1345.33325
                b = 0.52109685
                di = 2.3316e-04
                threshblur = qi/((1.0+b*di*t)**(1.0/max(b, 1.e-50)))

                if notblur < 0.8*threshblur:
                    continue

                '''
                #Face straight
                cnt = landmark.reshape(5,2)

                leftEye = cnt[0]
                rightEye = cnt[1]
                Nose = cnt[2]
                lefMouth = cnt[3]
                rightMouth = cnt[4]

                distLEye2Nose = np.linalg.norm(leftEye - Nose)
                distREye2Nose = np.linalg.norm(rightEye - Nose)
                disEyeMouth = np.linalg.norm((leftEye + rightEye)/2 - (lefMouth + rightMouth)/2)
                disTwoEye =  np.linalg.norm(leftEye - rightEye)

                verticalStraight = 2*abs(distLEye2Nose-distREye2Nose)/(distLEye2Nose+distREye2Nose)
                hoticalStraight = disTwoEye/disEyeMouth
                print(verticalStraight, hoticalStraight)
                if verticalStraight > 0.7 or hoticalStraight < 0.2 or hoticalStraight > 2.0  :
                    continue
                '''

                bbox_keeps.append(bbox)
                landmark_keeps.append(landmark)

            # Keeped
            names = []
            similarities = []
            for face_keypoints in landmark_keeps:
                face = share_param.system.sdk.align_face(image, face_keypoints)
                share_param.recog_lock.acquire()
                descriptor = share_param.system.sdk.get_descriptor(face)
                indicies, distances = share_param.system.sdk.find_most_similar(descriptor)
                share_param.recog_lock.release()
                user_name = share_param.system.get_user_name(indicies[0])
                names.append(user_name)
                similarities.append(distances[0])

            # if len(bboxes) > 0:
            #     print(bboxes)
            #     print(landmarks)
            #     print(names)
            #     print(similarities)
            for bbox, landmark, score, name in zip(bbox_keeps, landmark_keeps, similarities, names):
                boxes.append(bbox)
                points.append(landmark)
                scores.append(float(score))
                staffIds.append(name)
                faceIds.append(share_param.face_infos[name]["FaceId"]
                               if name in share_param.face_infos else -1)
                faceCropList.append(image[int(bbox[1]):int(
                    bbox[3]), int(bbox[0]):int(bbox[2])])

                faceW = abs(bbox[2] - bbox[0])
                faceH = abs(bbox[3] - bbox[1])
                expandLeft = max(0, bbox[0] - faceW/3)
                expandTop = max(0, bbox[1] - faceH/3)
                expandRight = min(bbox[2] + faceW/3, image.shape[1])
                expandBottom = min(bbox[3] + faceH/3, image.shape[0])
                faceCropExpandList.append(image[int(expandTop):int(
                    expandBottom), int(expandLeft):int(expandRight)])

                boxframeID.append(frameID)
                boxDeviceID.append(deviceId)
        if len(boxes) == 0:
            for i, frame in enumerate(frameList):
                custom_imshow(str(deviceIdList[i]),
                              cv2.resize(frame, (640, 480)))
            # print("TotalTime:",time.time() - totalTime)
            continue

        boxes = (np.array(boxes)/small_scale)
        points = (np.array(points)/small_scale)

        # print("boxes", boxes)
        # print("points", points)

        preTime = time.time()

        # print(scores)
        # print(index)
        for i, (box, staffId, faceId, score) in enumerate(zip(boxes, staffIds, faceIds, scores)):
            if score > support.get_dev_config()["DEV"]["face_reg_score"]:
                cv2.rectangle(frameList[boxframeID[i]], (int(box[0]), int(
                    box[1])), (int(box[2]), int(box[3])), (0, 255, 0), 2)
                y = box[1] - 15 if box[1] - 15 > 15 else box[1] + 15
                cv2.putText(frameList[boxframeID[i]], "{} {:03.3f}".format(staffId, score), (int(box[0]), int(y)), cv2.FONT_HERSHEY_SIMPLEX,
                            0.75, (0, 255, 0), 2)
            else:
                faceId = -1
                cv2.rectangle(frameList[boxframeID[i]], (int(box[0]), int(
                    box[1])), (int(box[2]), int(box[3])), (0, 0, 255), 2)
                y = box[1] - 15 if box[1] - 15 > 15 else box[1] + 15
                cv2.putText(frameList[boxframeID[i]], "{} {:03.3f}".format(staffId, score), (int(box[0]), int(y)), cv2.FONT_HERSHEY_SIMPLEX,
                            0.75, (0, 0, 255), 2)

            add_object_queue(
                staffId, faceId, boxDeviceID[i], datetime.datetime.now(), faceCropExpandList[i])

        for i, frame in enumerate(frameList):
            custom_imshow(str(deviceIdList[i]), cv2.resize(frame, (640, 480)))

        # print("TotalTime:",time.time() - totalTime)


def add_object_queue(staffId: str, face_id: int, device_id: str, track_time, face_img: np.array):
    data = {'EventId': "1",
            'DeviceId': device_id,
            "StaffId": staffId,
            'FaceId': face_id,
            'RecordTime': track_time.strftime("%Y-%m-%d %H:%M:%S"),
            'FaceImg': face_img}
    # print(data["RecordTime"])
    preTime = time.time()
    # print("share_param.batch_size:",share_param.batch_size)
    while share_param.object_queue.qsize() > 10*share_param.batch_size:
        share_param.object_queue.get()
    # print("RemTime:", time.time()-preTime)
    preTime = time.time()
    # print("QueueSize:", share_param.object_queue.qsize())
    if share_param.object_queue.qsize() < 5*share_param.batch_size:
        share_param.object_queue.put(data)
    else:
        if face_id != -1:
            share_param.object_queue.put(data)
    # print("PutTime:", time.time()-preTime)
    preTime = time.time()


def pushserver_thread_fun():
    url = support.create_url("face_upload")
    print("Full", url)
    lastTimeFaceID = {}
    while share_param.bRunning:
        time.sleep(0.001)
        if share_param.object_queue.empty():
            # print("object_queue empty")
            continue

        object_data = share_param.object_queue.get()

        data = {'EventId': object_data['EventId'],
                'DeviceId': object_data['DeviceId'],
                'RecordTime': object_data['RecordTime'],
                'FaceId': object_data['FaceId']}

        preTime = time.time()
        if object_data["FaceImg"] is None or object_data["FaceImg"].size == 0:
            continue

        object_data["FaceImg"] = cv2.cvtColor(
            object_data["FaceImg"], cv2.COLOR_RGB2BGR)
        _, buf = cv2.imencode(".jpg", object_data["FaceImg"])
        # print("encode time: {}".format(time.time()-preTime))
        filename = datetime.datetime.now().strftime("%Y%m%d%H%M%S%f") + ".jpg"

        if object_data['FaceId'] == -1:
            pathfile = "dataset/unknowns/" + filename
            path = os.path.dirname(pathfile)
            if not os.path.exists(path):
                os.makedirs(path)
            with open(pathfile, "wb") as f:
                f.write(buf)
            # continue
        else:
            pathfile = "dataset/knowns/" + \
                object_data["StaffId"] + "/" + filename
            path = os.path.dirname(pathfile)
            if not os.path.exists(path):
                os.makedirs(path)
            with open(pathfile, "wb") as f:
                f.write(buf)

        file = {"Files": (filename, buf.tobytes(),
                          "image/jpeg", {"Expires": "0"})}
        preTime = time.time()
        try:
            if data["FaceId"] not in lastTimeFaceID or (time.time() - lastTimeFaceID[data["FaceId"]]) > 10.0:
                lastTimeFaceID[data["FaceId"]] = time.time()
                # print("sending DeviceId: {},FaceId: {}".format(object_data["DeviceId"], object_data["FaceId"]))
                requests.post(url, files=file, params=data, timeout=3)
        except requests.exceptions.HTTPError as errh:
            print("Http Error:", errh)
        except requests.exceptions.ConnectionError as errc:
            print("Error Connecting:", errc)
        except requests.exceptions.Timeout as errt:
            print("Timeout Error:", errt)
        except requests.exceptions.RequestException as err:
            print("OOps: Something Else", err)
        except:
            print("OOps: Post error")
        # print("post time: {}".format(time.time()-preTime))


if __name__ == '__main__':
    share_param.cam_infos, share_param.face_infos = initiation()
    share_param.batch_size = len(share_param.cam_infos)
    share_param.stream_queue = queue.Queue(maxsize=15*share_param.batch_size)
    share_param.object_queue = queue.Queue(maxsize=15*share_param.batch_size)
    stream_threads = []
    for deviceID, camURL in share_param.cam_infos.items():
        stream_threads.append(threading.Thread(
            target=stream_thread_fun_oneCam, daemon=True, args=(deviceID, camURL)))

    tracking_thread = threading.Thread(target=tracking_thread_fun, daemon=True, args=())
    pushserver_thread = threading.Thread(target=pushserver_thread_fun, daemon=True, args=())
    fileserver = socketserver.TCPServer((share_param.devconfig["FILESERVER"]["host"], share_param.devconfig["FILESERVER"]["port"]), http.server.SimpleHTTPRequestHandler)
    file_thread = threading.Thread(target=fileserver.serve_forever, daemon=True, args=())
    
    share_param.bRunning = True
    file_thread.start()
    for stream_thread in stream_threads:
        stream_thread.start()
    tracking_thread.start()
    pushserver_thread.start()

    uvicorn.run(app, host=share_param.devconfig["APISERVER"]["host"], port=share_param.devconfig["APISERVER"]["port"])
    share_param.bRunning = False
    fileserver.shutdown()
    cv2.destroyAllWindows()
