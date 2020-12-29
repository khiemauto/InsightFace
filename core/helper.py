import base64
from io import StringIO

import cv2
import numpy as np
from PIL import Image

from core.config_util import get_config


def create_url(option):
    """Create url connect server"""
    # get info config
    ip = get_config('SERVER', 'ip')
    print('ip', ip)
    port = get_config('SERVER', 'port')
    print('port', port)
    url = get_config('URL', option)
    print('url', url)
    # create url
    if port:
        url = f"http://{ip}:{port}{url}"
    else:
        url = f"http://{ip}{url}"
    return url


def readb64(base64_string):
    """covert base64 image to cv2 image"""
    sbuf = StringIO()
    sbuf.write(base64.b64decode(base64_string))
    pimg = Image.open(sbuf)
    return cv2.cvtColor(np.array(pimg), cv2.COLOR_RGB2BGR)
