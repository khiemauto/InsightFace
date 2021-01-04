import queue, cv2, argparse, pickle, time, datetime, math, threading,requests, json, torch, os
from typing import Tuple
import numpy as np
from api.controler import config_controller
from core.helper import create_url
from flask import Flask
from PIL import Image
from face_recognition_sdk.utils.database import FaceRecognitionSystem
from api import web_server_interface

INDEX_DEVICE_ID = 0
INDEX_CAM_URL = 1
INDEX_VIDEOCAPTURE = 1


GET_FACE_INFO_URL = 'get_face_info'
GET_FACE_INFO_FILE = 'face_info.json'

GET_LIST_DEVICE_URL = 'get_list_device'
GET_LIST_DEVICE_FILE = 'list_device.json'

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", default="rtsp://admin:CongNghe@192.168.1.126:554/Streaming/Channels/101",
    help="path to input video")
ap.add_argument("-ln", "--label_name", default="output/label_name.json",
    help="path to label name")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
    help="confidence")
ap.add_argument("-fp", "--folders_path", default=None,
    help="path to save folders with images")
ap.add_argument("-dbp", "--db_folder_path", default="../employees/database",
    help="path to save database")
args = vars(ap.parse_args())

#Load recoginer and label
# le = pickle.loads(open("output/labelencoder.pickle", "rb").read())

system = FaceRecognitionSystem()

folders_path = args["folders_path"]
db_folder_path = args["db_folder_path"]

# create, save and load database initialized from folders containing user photos
if folders_path is not None:
    system.create_database_from_folders(folders_path)
    system.save_database(db_folder_path)
system.load_database(db_folder_path)

batch_size = 1
stream_queue = queue.Queue(maxsize=15*batch_size)
object_queue = queue.Queue(maxsize=15*batch_size)

cam_infos = {} 
face_infos = {}

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('Running on device: {}'.format(device))

def initiation() -> Tuple[dict, dict]:
    camera_datas = web_server_interface.get_infor(GET_LIST_DEVICE_URL, GET_LIST_DEVICE_FILE)
    face_datas =  web_server_interface.get_infor(GET_FACE_INFO_URL, GET_FACE_INFO_FILE)
    cam_dicts = {}
    face_dicts = {}
    for camera_data in camera_datas:
        cam_dicts[camera_data['DeviceId']] = camera_data['LinkRTSP']

    for face_data in face_datas:
        face_dicts[face_data["StaffCode"]] = face_data

    return cam_dicts, face_dicts

def stream_thread_fun_oneCam(deviceID: int, camURL: str):
    global stream_queue
    deviceId = deviceID
    cap = cv2.VideoCapture(camURL, cv2.CAP_FFMPEG)
    if cap is None or not cap.isOpened():
        print("[Error] Can't connect to {}".format(camURL))
        return

    FrameID = 1
    timeStep = 1/10
    preStep = time.time()

    while True:
        time.sleep(0.01)
        FrameID += 1
        if time.time() - preStep > timeStep:
            preStep = time.time()
            (grabbed, frame) = cap.retrieve()
            if not grabbed:
                continue
            if frame is None or frame.size == 0:
                continue
        else:
            cap.grab()
            continue

        while stream_queue.qsize() > 5*batch_size:
            stream_queue.get()
        stream_queue.put([deviceId, frame])

def tracking_thread_fun():
    global cam_infos, face_infos
    small_scale = 1
    while True:
        time.sleep(0.001)
        totalTime = time.time()
        if stream_queue.qsize() < batch_size:
            continue
        frameList = []      #[DeviceID, Image]
        rgbList=[]          #[DeviceID, Image]
        small_rgbList=[]    #[DeviceID, Image]
        preTime = time.time()

        deviceIdList = []
        for i in range(batch_size):
            deviceId, frame = stream_queue.get()
            xstart = (frame.shape[1] - frame.shape[0])//2
            frame = frame[:, xstart: xstart + frame.shape[0]]

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            small_rgb = cv2.resize(rgb, (0,0), fx=small_scale, fy=small_scale)
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

        preTime= time.time()

        for frameID, (image, deviceId) in enumerate(zip(small_rgbList, deviceIdList)):
            bboxes, landmarks = system.sdk.detect_faces(image)
            bbox_keeps = []
            landmark_keeps = []
            for bbox, landmark in zip(bboxes, landmarks):
                #Skip blur face
                imgcrop = image[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]
                if imgcrop is None or imgcrop.size == 0:
                    continue
                notblur = cv2.Laplacian(imgcrop, cv2.CV_32F).var()
                
                #Hyperbol blur
                faceW = abs(bbox[2] - bbox[0])
                faceH = abs(bbox[3] - bbox[1])
                # print(faceW*faceH, notblur)
                t = faceW*faceH
                qi = 1345.33325
                b = 0.52109685
                di =  2.3316e-04
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

            #Keeped
            user_ids = []
            similarities = []

            for face_keypoints in landmark_keeps:
                face = system.sdk.align_face(image, face_keypoints)
                # cv2.imshow("face", face)
                descriptor = system.sdk.get_descriptor(face)
                indicies, distances = system.sdk.find_most_similar(descriptor)
                user_ids.append(indicies[0])
                similarities.append(distances[0])

            # bboxes, landmarks, user_ids, similarities = system.sdk.recognize_faces(small_rgbList[i])
            names = [system.get_user_name(uid) for uid in user_ids]
            
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
                faceIds.append(face_infos[name]["FaceId"] if name in face_infos else -1)
                faceCropList.append(image[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])])

                faceW = abs(bbox[2] - bbox[0])
                faceH = abs(bbox[3] - bbox[1])
                expandLeft = max(0, bbox[0] - faceW/3)
                expandTop = max(0, bbox[1] - faceH/3)
                expandRight = min(bbox[2] + faceW/3, image.shape[1])
                expandBottom = min(bbox[3] + faceH/3, image.shape[0])
                faceCropExpandList.append(image[int(expandTop):int(expandBottom), int(expandLeft):int(expandRight)])

                boxframeID.append(frameID)
                boxDeviceID.append(deviceId)
        
        if len(boxes) == 0:
            for i, frame in enumerate(frameList):
                cv2.imshow(str(deviceIdList[i]), cv2.resize(frame,(640,480)))
                cv2.waitKey(1)    
            # print("TotalTime:",time.time() - totalTime)
            continue
        
        boxes = (np.array(boxes)/small_scale)
        points = (np.array(points)/small_scale)

        # print("boxes", boxes)
        # print("points", points)

        preTime= time.time()

        # print(scores)
        # print(index)
        for i, (box, staffId, faceId, score) in enumerate(zip(boxes, staffIds, faceIds, scores)):
            if score > 0.6:
                cv2.rectangle(frameList[boxframeID[i]], (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 255, 0), 2)
                y = box[1] - 15 if box[1] - 15 > 15 else box[1] + 15
                cv2.putText(frameList[boxframeID[i]], "{} {:03.3f}".format(staffId, score), (int(box[0]), int(y)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.75, (0, 255, 0), 2)
            else:
                faceId = -1
                cv2.rectangle(frameList[boxframeID[i]], (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 0, 255), 2)
                y = box[1] - 15 if box[1] - 15 > 15 else box[1] + 15
                cv2.putText(frameList[boxframeID[i]], "{} {:03.3f}".format(staffId, score), (int(box[0]), int(y)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.75, (0, 0, 255), 2)
            
            add_object_queue(staffId, faceId, boxDeviceID[i], datetime.datetime.now(), faceCropExpandList[i])

        for i, frame in enumerate(frameList):
            cv2.imshow(str(deviceIdList[i]), cv2.resize(frame,(640,480)))
            cv2.waitKey(1)

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
    # print("batch_size:",batch_size)
    while object_queue.qsize() > 10*batch_size:
        object_queue.get()
    # print("RemTime:", time.time()-preTime)
    preTime = time.time()
    # print("QueueSize:", object_queue.qsize())
    if object_queue.qsize() < 5*batch_size:
        object_queue.put(data)
    else:
        if face_id != -1:
            object_queue.put(data)
    # print("PutTime:", time.time()-preTime)
    preTime = time.time()
    
def pushserver_thread_fun():
    url = create_url("face_upload")
    print("Full", url)
    lastTimeFaceID = {}
    while True:
        time.sleep(0.001)
        if object_queue.empty():
            # print("object_queue empty")
            continue

        object_data = object_queue.get()
    
        data = {'EventId': object_data['EventId'],
                'DeviceId': object_data['DeviceId'],
                'RecordTime': object_data['RecordTime'],
                'FaceId': object_data['FaceId']}

        preTime = time.time()
        if object_data["FaceImg"] is None or object_data["FaceImg"].size == 0:
            continue

        object_data["FaceImg"] = cv2.cvtColor(object_data["FaceImg"], cv2.COLOR_RGB2BGR)
        _, buf = cv2.imencode(".jpg", object_data["FaceImg"])
        # print("encode time: {}".format(time.time()-preTime))
        filename = datetime.datetime.now().strftime("%Y%m%d%H%M%S%f") + ".jpg"

        if object_data['FaceId'] == -1:
            pathfile = "dataset/unknows/" + filename
            path = os.path.dirname(pathfile)
            if not os.path.exists(path):
                os.makedirs(path)
            with open(pathfile, "wb") as f:
                f.write(buf)
            # continue
        else:
            pathfile = "dataset/knows/" + object_data["StaffId"] + "/" + filename
            path = os.path.dirname(pathfile)
            if not os.path.exists(path):
                os.makedirs(path)
            with open(pathfile, "wb") as f:
                f.write(buf)

        file = {"Files": (filename, buf.tobytes(), "image/jpeg", {"Expires": "0"})}
        preTime = time.time()
        try:
            if data["FaceId"] not in lastTimeFaceID or (time.time() - lastTimeFaceID[data["FaceId"]]) > 10.0:
                lastTimeFaceID[data["FaceId"]] = time.time()
                # print("sending DeviceId: {},FaceId: {}".format(object_data["DeviceId"], object_data["FaceId"]))
                requests.post(url, files=file, params=data, timeout=3)
        except requests.exceptions.HTTPError as errh:
            print ("Http Error:",errh)
        except requests.exceptions.ConnectionError as errc:
            print ("Error Connecting:",errc)
        except requests.exceptions.Timeout as errt:
            print ("Timeout Error:",errt)
        except requests.exceptions.RequestException as err:
            print ("OOps: Something Else",err)
        except:
            print ("OOps: Post error")
        # print("post time: {}".format(time.time()-preTime))

if __name__ == '__main__':
    cam_infos, face_infos = initiation()
    batch_size = len(cam_infos)
    stream_queue = queue.Queue(maxsize=15*batch_size)
    object_queue = queue.Queue(maxsize=15*batch_size)
    stream_threads = []
    for deviceID, camURL in cam_infos.items():
        stream_threads.append(threading.Thread(target=stream_thread_fun_oneCam, args=(deviceID, camURL)))

    tracking_thread = threading.Thread(target=tracking_thread_fun, args=())
    pushserver_thread = threading.Thread(target=pushserver_thread_fun, args=())

    for stream_thread in stream_threads:
        stream_thread.start()
    
    tracking_thread.start()
    pushserver_thread.start()
    for stream_thread in stream_threads:
        stream_thread.join()
    tracking_thread.join()
    pushserver_thread.join()