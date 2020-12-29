import queue, cv2, argparse, imutils, pickle, time, datetime, math, threading,requests, json, torch, os
import numpy as np
from face_align import FaceAlignerDlib
from api.controler import config_controller
from core.helper import create_url
from flask import Flask

from torch.nn import functional as F
from facenet_pytorch import MTCNN, InceptionResnetV1, fixed_image_standardization, training
from torch.utils.data import DataLoader, SubsetRandomSampler
from torch import optim
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms
from PIL import Image
import dlib
from imutils import face_utils

from typing import List, Tuple, Dict
from imutils.video import WebcamVideoStream


INDEX_DEVICE_ID = 0
INDEX_CAM_URL = 1
INDEX_VIDEOCAPTURE = 1

ap = argparse.ArgumentParser()
# ap.add_argument("-i", "--input", default="rtsp://admin:CongNghe@192.168.1.126:554/Streaming/Channels/101",
#     help="path to input video")
ap.add_argument("-i", "--input", default="rtsp://admin:CongNghe@192.168.1.126:554/Streaming/Channels/101",
# ap.add_argument("-i", "--input", default="videos/test.mp4",
    help="path to input video")
ap.add_argument("-p", "--shape_predictor", default="model/shape_predictor_5_face_landmarks.dat",
    help="path to facial landmark predictor")
ap.add_argument("-r", "--recognizer", default="output/resnet.pth",
    help="path to classifite model")
ap.add_argument("-l", "--label", default="output/label.json",
    help="path to label")
ap.add_argument("-ln", "--label_name", default="output/label_name.json",
    help="path to label name")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
    help="confidence")
args = vars(ap.parse_args())

batch_size = 2

stream_queue = queue.Queue(maxsize=15*batch_size)
object_queue = queue.Queue(maxsize=15*batch_size)

print("Loading label: ", args["label"])
with open(args["label"]) as json_file:
    label = json.load(json_file)

with open(args["label_name"]) as json_file:
    label_name = json.load(json_file)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('Running on device: {}'.format(device))

detector = dlib.cnn_face_detection_model_v1("model/mmod_human_face_detector.dat")
predictor = dlib.shape_predictor(args["shape_predictor"])
fa = FaceAlignerDlib(faceSize=(160,160))

trans = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((160,160)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

softmax = torch.nn.Softmax(dim=1)

print("Loading resnet: ", args["recognizer"])
resnet = torch.load(args["recognizer"], map_location=device)
resnet.eval()

fourcc = cv2.VideoWriter_fourcc(*"MJPG")
writer = cv2.VideoWriter("output/test.avi", fourcc, 20, (1920, 1080), True)
# torch.set_grad_enabled(False)

def initiation():
    camera_datas = config_controller.get_config_device()
    ret = []
    for camera_data in camera_datas:
        device_id = camera_data['DeviceId']
        cam_url = camera_data['LinkRTSP']
        ret.append([device_id, cam_url])
    
    return ret

def stream_thread_fun(cam_infors):
    global stream_queue
    streams = []    #[DeviceID, VideoCapture]
    print(cam_infors)
    for cam_infor in cam_infors:
        print(cam_infor[INDEX_CAM_URL])
        vs = WebcamVideoStream(src=cam_infor[INDEX_CAM_URL]).start()

        cap = cv2.VideoCapture(cam_infor[INDEX_CAM_URL], cv2.CAP_FFMPEG)
        if cap is None or not cap.isOpened():
            print("[Error] Can't connect to {}".format(cam_infor[INDEX_CAM_URL]))
            continue

        streams.append([cam_infor[INDEX_DEVICE_ID], cap])
    
    FrameID = 1

    while True:
        time.sleep(0.001)
        #TEST
        # if stream_queue.qsize() > 8*batch_size:
        #     continue
        #
        FrameID += 1
        for stream in streams:
            # print(stream[INDEX_DEVICE_ID])
            if FrameID%3 == 0:
                # stream[INDEX_VIDEOCAPTURE].grab()
                (grabbed, frame) = stream[INDEX_VIDEOCAPTURE].read()
                if not grabbed:
                    continue
            else:
                stream[INDEX_VIDEOCAPTURE].grab()
                continue

            # if FrameID%2 != 0:
            #     continue

            while stream_queue.qsize() > 5*batch_size:
                stream_queue.get()
            stream_queue.put([stream[INDEX_DEVICE_ID], frame])


def stream_thread_fun_oneCam(cam_infor):
    global stream_queue
    deviceId = cam_infor[INDEX_DEVICE_ID]
    cap = cv2.VideoCapture(cam_infor[INDEX_CAM_URL], cv2.CAP_FFMPEG)
    if cap is None or not cap.isOpened():
        print("[Error] Can't connect to {}".format(cam_infor[INDEX_CAM_URL]))
        return

    FrameID = 1

    while True:
        time.sleep(0.001)
        FrameID += 1
        if FrameID%3 == 0:
            (grabbed, frame) = cap.retrieve()
            if not grabbed:
                continue
        else:
            cap.grab()
            continue

        while stream_queue.qsize() > 5*batch_size:
            stream_queue.get()
        stream_queue.put([deviceId, frame])


def tracking_thread_fun():
    mtcnn_scale = 1
    # resnet.eval()

    # for param in resnet.parameters():
    #     param.requires_grad = False
        # print(param.requires_grad)
    while True:
        time.sleep(0.001)
        totalTime = time.time()
        if stream_queue.qsize() < batch_size:
            continue
        frameList = []      #[DeviceID, Image]
        rgbList=[]          #[DeviceID, Image]
        small_rgbList=[]    #[DeviceID, Image]
        # rgb_pilList=[]
        # small_rgb_pilList=[]
        preTime = time.time()

        deviceIdList = []
        for i in range(batch_size):
            deviceId, frame = stream_queue.get()
            xstart = (frame.shape[1] - frame.shape[0])//2
            frame = frame[:, xstart: xstart + frame.shape[0]]

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            small_rgb = cv2.resize(rgb, (0,0), fx=mtcnn_scale, fy=mtcnn_scale)
            # rgb_pil = Image.fromarray(rgb)
            # small_rgb_pil = Image.fromarray(small_rgb)
            frameList.append(frame)
            rgbList.append(rgb)
            small_rgbList.append(small_rgb)
            deviceIdList.append(deviceId)
            # rgb_pilList.append(rgb_pil)
            # small_rgb_pilList.append(small_rgb_pil)

        # print("Resize and convert time:", time.time()- preTime)

        preTime= time.time()
        dets = detector(small_rgbList, 0, batch_size = batch_size)
        # print("CNN_Detect time:", time.time()- preTime)
        # print("CNN Frame len", len(dets))

        boxDeviceID = []
        boxframeID = []
        boxes = []
        probs = []
        points = []

        # print("dets", dets)

        for i in range(len(dets)):
            if dets[i] is not None:
                # print(dets[i])
                for det in dets[i]:
                    # print("rect", det.rect)
                    # print("score", det.confidence)
                    # print(det.rect.right() - det.rect.left(), det.rect.bottom() - det.rect.top())
                    if det is not None and det.confidence > 1.0:
                        if (det.rect.left() < 10 or det.rect.right() > small_rgbList[i].shape[1] - 10
                            or det.rect.top() < 10 or det.rect.bottom() > small_rgbList[i].shape[0] - 10):
                            continue

                        if det.rect.right() - det.rect.left() < 80 or det.rect.bottom() - det.rect.top() < 80:
                            continue
                        imgcrop = small_rgbList[i][det.rect.top(): det.rect.bottom(), det.rect.left(): det.rect.right()]
                        if imgcrop.size == 0:
                            continue
                        notblur = cv2.Laplacian(imgcrop, cv2.CV_32F).var()
                        if notblur < 120.0:
                            continue

                        boxes.append([det.rect.left(), det.rect.top(), det.rect.right(), det.rect.bottom()])
                        probs.append(det.confidence)
                        shape = predictor(small_rgbList[i], det.rect)
                        shape = face_utils.shape_to_np(shape)
                        points.append(shape)
                        boxframeID.append(i)
                        boxDeviceID.append(deviceIdList[i])
        
        if len(boxes) == 0:
            for i, frame in enumerate(frameList):
                cv2.imshow(str(deviceIdList[i]), cv2.resize(frame,(640,480)))
                cv2.waitKey(1)    
                # writer.write(frame)
            continue
        
        boxes = (np.array(boxes)/mtcnn_scale)
        points = (np.array(points)/mtcnn_scale)

        # print("boxes", boxes)
        # print("points", points)

        preTime= time.time()
        tensorsList = []

        faceAlignedList = []

        for i in range(len(boxes)):
            faceAligned, _ = fa.align(rgbList[boxframeID[i]], boxes[i], points[i])
            faceAlignedList.append(faceAligned)
            # cv2.imshow(str(i), faceAligned)
            # cv2.waitKey(1)
            imgTensor = trans(faceAligned)
            tensorsList.append(imgTensor)

        print("Align time:", time.time()- preTime)

        preTime= time.time()
        inputTensor = torch.stack(tensorsList).to(device)
        preTime= time.time()
        # print(inputTensor)
        with torch.no_grad():
            outputTensor = resnet(inputTensor)

        print("RESNET time:", time.time()- preTime)

        names = []
        scores = []
        faceIds = []
        staffIds = []

        for score in outputTensor:
            index = int(torch.argmax(score))
            scores.append(float(score[index]))
            staffIds.append(label[str(index)])
            names.append(label_name[label[str(index)]]["FaceName"])
            faceIds.append(label_name[label[str(index)]]["FaceId"])
        
        print(scores)
        print(index)
        # loop over the recognized faces
        for i, (box, staffId, name, faceId, score) in enumerate(zip(boxes, staffIds, names, faceIds, scores)):
            # draw the predicted face name on the image
            # print("FrameID", boxframeID[i])
            # print("BOX", box)
            # print("Y", int(y))
            if score > 7.5:
                cv2.rectangle(frameList[boxframeID[i]], (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 255, 0), 2)
                y = box[1] - 15 if box[1] - 15 > 15 else box[1] + 15
                cv2.putText(frameList[boxframeID[i]], "{} {:03.3f}".format(name, score), (int(box[0]), int(y)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.75, (0, 255, 0), 2)
            else:
                faceId = -1
                cv2.rectangle(frameList[boxframeID[i]], (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 0, 255), 2)
                y = box[1] - 15 if box[1] - 15 > 15 else box[1] + 15
                cv2.putText(frameList[boxframeID[i]], "{} {:03.3f}".format(name, score), (int(box[0]), int(y)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.75, (0, 0, 255), 2)
            add_object_queue(staffId, faceId, boxDeviceID[i], datetime.datetime.now(), faceAlignedList[i])

        for i, frame in enumerate(frameList):
            cv2.imshow(str(deviceIdList[i]), cv2.resize(frame,(640,480)))
            cv2.waitKey(1)
        print("TotalTime:",time.time() - totalTime)
            # print("write")    
            # writer.write(frame)
            # cv2.imwrite(str(time.time())+ ".jpg", frame)

def add_object_queue(staffId: str, face_id: int, device_id: str, track_time, face_img: np.array):
    data = {'EventId': "1",
            'DeviceId': device_id,
            "StaffId": staffId,
            'FaceId': face_id,
            'RecordTime': track_time.strftime("%Y-%m-%d %H:%M:%S"),
            'FaceImg': face_img}
    print(data["RecordTime"])
    object_queue.put(data)
    while object_queue.qsize() > 10*batch_size:
            object_queue.get()
    
def pushserver_thread_fun():
    url = create_url("face_upload")
    print("Full", url)
    while True:
        time.sleep(0.001)
        if object_queue.empty():
            continue

        object_data = object_queue.get()
    
        data = {'EventId': object_data['EventId'],
                'DeviceId': object_data['DeviceId'],
                'RecordTime': object_data['RecordTime'],
                'FaceId': object_data['FaceId']}

        print("sending DeviceId: {},FaceId: {}".format(object_data["DeviceId"], object_data["FaceId"]))
        preTime = time.time()
        object_data["FaceImg"] = cv2.cvtColor(object_data["FaceImg"], cv2.COLOR_RGB2BGR)
        _, buf = cv2.imencode(".png", object_data["FaceImg"])
        print("encode time: {}".format(time.time()-preTime))
        filename = datetime.datetime.now().strftime("%Y%m%d%H%M%S%f") + ".png"

        if object_data['FaceId'] == -1:
            pathfile = "dataset/unknows/" + filename
            path = os.path.dirname(pathfile)
            if not os.path.exists(path):
                os.makedirs(path)
            cv2.imwrite(pathfile, object_data["FaceImg"])
        else:
            pathfile = "dataset/knows/" + object_data["StaffId"] + "/" + filename
            path = os.path.dirname(pathfile)
            if not os.path.exists(path):
                os.makedirs(path)
            cv2.imwrite(pathfile, object_data["FaceImg"])

        file = {"Files": (filename, buf.tobytes(), "image/png", {"Expires": "0"})}
        preTime = time.time()
        # r = requests.post(url, files=file, params=data)
        print("post time: {}".format(time.time()-preTime))

        # print(r.status_code)
        # if r.status_code == 201:
        # print(r.content)

if __name__ == '__main__':
    # set_up_box_weak_up()
    cam_infors = initiation()
    batch_size = len(cam_infors)
    stream_queue = queue.Queue(maxsize=15*batch_size)
    object_queue = queue.Queue(maxsize=15*batch_size)
    stream_threads = []
    for cam_infor in cam_infors:
        stream_threads.append(threading.Thread(target=stream_thread_fun_oneCam, args=(cam_infor,)))
    # stream_thread = threading.Thread(target=stream_thread_fun, args=(args["input"],))
    # stream_thread = threading.Thread(target=stream_thread_fun, args=(cam_infors,))
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