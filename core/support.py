import json
import requests
import cv2
import numpy as np
import share_param

def get_blur_var(area: float) -> float:
    return share_param.qi/((1.0+share_param.b*share_param.di*area)**(1.0/max(share_param.b, 1.e-50)))

def get_dev_config(local_file = "devconfig.json") -> json:
    """
    Get deverlop config
    :local_file: dev config file
    :return: json config
    """
    try:
        get_dev_config.camera_data
    except AttributeError or NameError:
        get_dev_config.camera_data = None
        try:
            with open(local_file, 'r') as json_file:
                get_dev_config.camera_data = json.load(json_file)
        except:
            print(f"[Error] Read json {local_file}")
    return get_dev_config.camera_data

def create_url(option: str):
    config = share_param.devconfig
    if config is None:
        return None
    ip = config['SERVER']['ip']
    port = config['SERVER']['port']
    url = config['URL'][option]
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
    if get_dev_config()["DEV"]["imshow"]:
        cv2.imshow(title, image)
        cv2.waitKey(1)