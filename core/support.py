import json
import requests
import cv2
import numpy as np
from core import share_param

def get_blur_var(area: float) -> float:
    return share_param.qi/((1.0+share_param.b*share_param.di*area)**(1.0/max(share_param.b, 1.e-50)))

def get_dev_config(local_file = "devconfig.json") -> json:
    """
    Get deverlop config
    :local_file: dev config file
    :return: json config
    """
    camera_data = None
    try:
        with open(local_file, 'r') as json_file:
            camera_data = json.load(json_file)
    except:
        print(f"[Error] Read json {local_file}")
    return camera_data

def create_url(option: str):
    config = share_param.devconfig
    if config is None:
        return None
    ip = config['SERVER']['ip']
    port = config['SERVER']['port']
    url = config['SERVER'][option]
    if port:
        url = f"http://{ip}:{port}{url}"
    else:
        url = f"http://{ip}{url}"
    print('[INFO] URL', url)
    return url

def get_infor(url_name: str, local_file: str) -> json:
    url = create_url(url_name)
    if not url:
        return None
    try:
        response = requests.get(url)
        if response.status_code == 200:
            with open(local_file, 'w') as json_file:
                json.dump(response.json(), json_file)
    except:
        print("[Error] Get info from {url_name}")
    camera_data = None
    try:
        with open(local_file, 'r') as json_file:
            camera_data = json.load(json_file)
    except:
        print("[Error] Get info from {json_file}")

    return camera_data

def get_cameras() -> dict:
    cam_dicts = {}
    camera_datas = get_infor(share_param.GET_LIST_DEVICE_URL, share_param.GET_LIST_DEVICE_FILE)
    for camera_data in camera_datas:
        cam_dicts[camera_data['DeviceId']] = camera_data['LinkRTSP']
    return cam_dicts

def get_faces() -> dict:
    face_dicts = {}
    face_datas = get_infor(share_param.GET_FACE_INFO_URL, share_param.GET_FACE_INFO_FILE)
    for face_data in face_datas:
        face_dicts[face_data["StaffCode"]] = face_data
    return face_dicts


def custom_imshow(title: str, image: np.ndarray):
    if share_param.devconfig["DEV"]["imshow"]:
        cv2.imshow(title, image)
        cv2.waitKey(1)

def add_stream_queue(data):
    while share_param.stream_queue.qsize() > share_param.STREAM_SIZE*share_param.batch_size:
            share_param.stream_queue.get()
    share_param.stream_queue.put(data)

def add_detect_queue(data):
    while share_param.detect_queue.qsize() > share_param.DETECT_SIZE*share_param.batch_size:
        share_param.detect_queue.get()
    share_param.detect_queue.put(data)

def add_push_detect_queue(data):
    while share_param.push_detect_queue.qsize() > share_param.DETECT_SIZE*share_param.batch_size:
        share_param.push_detect_queue.get()
    share_param.push_detect_queue.put(data)

def get_system_status() -> json:
    ret = {}
    ret["running"] = share_param.bRunning
    camera_status = {}
    for deviceID in share_param.stream_threads:
        camera_status[deviceID] = share_param.stream_threads[deviceID].is_alive()
    
    ret["camera"] = camera_status
    ret["detect"] = share_param.detect_thread.is_alive()
    ret["recogn"] = share_param.recogn_thread.is_alive()
    ret["pushs_server"] = share_param.pushserver_thread.is_alive()
    ret["file_sever"] = share_param.file_thread.is_alive()

    return ret