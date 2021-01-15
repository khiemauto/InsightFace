import share_param
import numpy as np
from core import support
import time, datetime, os, sys, cv2, requests

def add_object_queue(user_name: str, device_id: str, track_time: str , face_img: np.array):
    face_id = share_param.face_infos[user_name]["FaceId"] if user_name in share_param.face_infos else -1

    data = {'EventId': "1",
            'UserName': user_name,
            'DeviceId': device_id,
            'FaceId': face_id,
            'RecordTime': track_time,
            'FaceImg': face_img}
    while share_param.recogn_queue.qsize() > share_param.RECOGN_SIZE*share_param.batch_size:
        share_param.recogn_queue.get()
    if share_param.recogn_queue.qsize() < share_param.RECOGN_SIZE*share_param.batch_size//2:
        share_param.recogn_queue.put(data)
    elif face_id != -1:
        share_param.recogn_queue.put(data)

def pushserver_thread_fun():
    url = support.create_url("face_upload")
    print("Full", url)
    lastTimeFaceID = {}
    while share_param.bRunning:
        time.sleep(0.001)
        if share_param.recogn_queue.empty():
            continue

        object_data = share_param.recogn_queue.get()

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
                object_data["UserName"] + "/" + filename
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
                print("sending DeviceId: {},FaceId: {}".format(object_data["DeviceId"], object_data["FaceId"]))
                ret = requests.post(url, files=file, params=data, timeout=3)
                # print(data["FaceId"], ret.content)
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