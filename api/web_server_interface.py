import json
import os

import requests
from core.helper import create_url

def get_infor_from_server(url_name: str, local_file: str):
    url = create_url(url_name)
    try:
        response = requests.get(url)
        if response.status_code == 200:
            print(response)
            with open(local_file, 'w') as json_file:
                json.dump(response.json(), json_file)
    except:
        print("[Error] Get info from server")
        return

def get_infor_from_local(local_file: str):
    print(local_file)
    with open(local_file, 'r') as json_file:
        camera_data = json.load(json_file)
    return camera_data

def get_infor(url_name: str, local_file: str) -> json:
    try:
        get_infor_from_server(url_name, local_file)
    finally:
        camera_data = get_infor_from_local(local_file)

    return camera_data