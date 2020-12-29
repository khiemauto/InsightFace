import queue, cv2, argparse, imutils, pickle, time, datetime, math, threading,requests, json, torch, os
import numpy as np
from api.controler import config_controller
from core.helper import create_url
from flask import Flask

from torch.utils.data import DataLoader, SubsetRandomSampler
from torch import optim
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms
from PIL import Image
import dlib
from imutils import face_utils
from face_recognition_sdk.utils.database import FaceRecognitionSystem
from face_recognition_sdk.utils.io_utils import read_image, save_image
from face_recognition_sdk.utils.draw_utils import draw_boxes, draw_landmarks

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
ap.add_argument("-fp", "--folders_path", default=None,
    help="path to save folders with images")
ap.add_argument("-dbp", "--db_folder_path", default="../employees/database",
    help="path to save database")
args = vars(ap.parse_args())

#Load recoginer and label
recognizer = pickle.loads(open("output/recognizer.pickle", "rb").read())
le = pickle.loads(open("output/labelencoder.pickle", "rb").read())

system = FaceRecognitionSystem()

folders_path = args["folders_path"]
db_folder_path = args["db_folder_path"]

# create, save and load database initialized from folders containing user photos
if folders_path is not None:
    system.create_database_from_folders(folders_path)
    system.save_database(db_folder_path)

system.load_database(db_folder_path)


# print("embedded", list(embedded)[0])

batch_size = 1

stream_queue = queue.Queue(maxsize=15*batch_size)
object_queue = queue.Queue(maxsize=15*batch_size)

print("Loading label: ", args["label"])
with open(args["label"]) as json_file:
    label = json.load(json_file)

with open(args["label_name"]) as json_file:
    label_name = json.load(json_file)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('Running on device: {}'.format(device))

embedded = list(pickle.loads(open("output/embedded.pickle", "rb").read()))
# print(embedded)

detector = dlib.cnn_face_detection_model_v1("model/mmod_human_face_detector.dat")
predictor = dlib.shape_predictor(args["shape_predictor"])

# trans = transforms.Compose([
#     np.float32,
#     transforms.ToTensor(),
#     transforms.Resize((160,160)),
#     fixed_image_standardization
# ])
trans = transforms.Compose([
    # np.float32,
    transforms.ToPILImage(),
    transforms.Resize((160,160)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    # transforms.ToTensor(),
    # transforms.Resize((160,160)),
    # fixed_image_standardization
])

softmax = torch.nn.Softmax(dim=1)

print("Loading resnet: ", args["recognizer"])
# resnet = torch.load(args["recognizer"], map_location=device)
# resnet.eval()

fourcc = cv2.VideoWriter_fourcc(*"MJPG")
writer = cv2.VideoWriter("output/test.avi", fourcc, 20, (1920, 1080), True)
# torch.set_grad_enabled(False)

def initiation():
    camera_data = config_controller.get_config_device()
    # print(camera_data)
    data = camera_data[0]
    device_id = data['DeviceId']
    cam_url = data['LinkRTSP']
    return device_id, cam_url

def stream_thread_fun(cam_url: str):
    global stream_queue
    stream = cv2.VideoCapture(cam_url, cv2.CAP_FFMPEG)
    if not stream.isOpened():
        print("[Error] Can't connect to {}".format(cam_url))
        return
    
    FrameID = 1

    while True:
        time.sleep(0.001)
        #TEST
        # if stream_queue.qsize() > 8*batch_size:
        #     continue
        #
        (grabbed, frame) = stream.read()
        if not grabbed:
            # writer.release()
            continue

        FrameID += 1
        # if FrameID%5 != 0:
        #     continue

        while stream_queue.qsize() > 5*batch_size:
            stream_queue.get()
        stream_queue.put(frame)

def tracking_thread_fun(device_id: str):
    with torch.no_grad():
        mtcnn_scale = 1
        while True:
            time.sleep(0.001)
            if stream_queue.qsize() < batch_size:
                continue
            frameList = []
            rgbList=[]
            small_rgbList=[]
            # rgb_pilList=[]
            # small_rgb_pilList=[]
            preTime = time.time()
            for i in range(batch_size):
                frame = stream_queue.get()
                xstart = (frame.shape[1] - frame.shape[0])//2
                frame = frame[:, xstart: xstart + frame.shape[0]]

                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                small_rgb = cv2.resize(rgb, (0,0), fx=mtcnn_scale, fy=mtcnn_scale)
                # rgb_pil = Image.fromarray(rgb)
                # small_rgb_pil = Image.fromarray(small_rgb)
                frameList.append(frame)
                rgbList.append(rgb)
                small_rgbList.append(small_rgb)
                # rgb_pilList.append(rgb_pil)
                # small_rgb_pilList.append(small_rgb_pil)

            # print("Resize and convert time:", time.time()- preTime)

            

            boxframeID = []
            boxes = []
            points = []
            # names = []
            scores = []
            faceIds = []
            staffIds = []

            faceCropList = []

            preTime= time.time()

            for i in range(batch_size):
                bboxes, landmarks, user_ids, similarities = system.sdk.recognize_faces(small_rgbList[i])
                names = [system.get_user_name(uid) for uid in user_ids]
                
                if len(bboxes) > 0:
                    print(bboxes)
                    print(landmarks)
                    print(names)
                    print(similarities)

                for j in range(len(bboxes)):
                    boxes.append(bboxes[j])
                    points.append(landmarks[j])
                    scores.append(float(similarities[j]))
                    staffIds.append(names[j])
                    # names.append(names[j])
                    faceIds.append(label_name[names[j]]["FaceId"])
                    faceCropList.append(small_rgbList[i][int(bboxes[j][1]):int(bboxes[j][3]), int(bboxes[j][0]):int(bboxes[j][2])])
                    boxframeID.append(i)
            
            # print("Recogni time:", time.time() - preTime)
            if len(boxes) == 0:
                for frame in frameList:
                    cv2.imshow("frame", frame)
                    cv2.waitKey(1)    
                continue
            
            print(scores)
            # print(index)
            # loop over the recognized faces
            for i, (box, staffId, faceId, score) in enumerate(zip(boxes, staffIds, faceIds, scores)):
                # draw the predicted face name on the image
                # print("FrameID", boxframeID[i])
                # print("BOX", box)
                # print("Y", int(y))
                if score > 0.5:
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
                add_object_queue(staffId, faceId, device_id, datetime.datetime.now(), faceCropList[i])

            for frame in frameList:
                cv2.imshow("frame", frame)
                cv2.waitKey(1)
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
    preTime = time.time()
    print("batch_size:",batch_size)
    while object_queue.qsize() > 10*batch_size:
        object_queue.get()
    print("RemTime:", time.time()-preTime)
    preTime = time.time()
    print("QueueSize:", object_queue.qsize())
    if object_queue.qsize() < 5*batch_size:
        object_queue.put(data)
    else:
        if face_id != -1:
            object_queue.put(data)
    print("PutTime:", time.time()-preTime)
    preTime = time.time()
    
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
        r = requests.post(url, files=file, params=data)
        print("post time: {}".format(time.time()-preTime))

if __name__ == '__main__':
    # set_up_box_weak_up()
    device_id, cam_url = initiation()
    print (device_id)
    print (cam_url)
    # stream_thread = threading.Thread(target=stream_thread_fun, args=(args["input"],))
    stream_thread = threading.Thread(target=stream_thread_fun, args=(cam_url,))
    tracking_thread = threading.Thread(target=tracking_thread_fun, args=(device_id,))
    pushserver_thread = threading.Thread(target=pushserver_thread_fun, args=())

    stream_thread.start()
    tracking_thread.start()
    pushserver_thread.start()
    stream_thread.join()
    tracking_thread.join()
    pushserver_thread.join()