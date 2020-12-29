import json
import os

import requests

from core.helper import create_url

GET_CAMERA_RECOGNITION_URL = 'get_list_device'
FACE_DATA_FILE = 'config.json'


def get_config_from_server():
    """
    create config file from device info in database
    :return: None
    """
    url = create_url(GET_CAMERA_RECOGNITION_URL)
    response = requests.get(url)
    config = response.content.decode()
    if config:
        with open(FACE_DATA_FILE, 'wb') as json_file:
            json_file.write(config.encode())


def get_config_from_local():
    """
    create camera info from file config
    :return: list camera info
    """
    if os.path.getsize(FACE_DATA_FILE):
        with open(FACE_DATA_FILE, 'rb') as json_file:
            camera_data = json.load(json_file)
    return camera_data


def get_config_device():
    """
    create list camera info
    :return: list camera info
    """
    try:
        get_config_from_server()
    finally:
        camera_data = get_config_from_local()

    return camera_data
