import json
# from requests.api import get
import requests

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
    config = get_dev_config()
    if config is None:
        return None
    ip = config['SERVER']['ip']
    print('ip', ip)
    port = config['SERVER']['port']
    print('port', port)
    url = config['URL'][option]
    print('url', url)
    if port:
        url = f"http://{ip}:{port}{url}"
    else:
        url = f"http://{ip}{url}"
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