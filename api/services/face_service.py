import base64
import io
import json
import os
import time

import cv2
import face_recognition
import requests
from PIL import Image

from api.services.face_recognition_service import face_encode
from core.helper import create_url
from model.face_recognition_model import FaceRecognitionModel, save_known_faces, load_known_faces

face_recognition_model = FaceRecognitionModel.get_instance()

GET_FACE_IMAGE_URL = 'get_face_image'
DOWNLOAD_IMAGE_URL = 'download_image'
GET_IMAGE_ID_URL = 'get_list_image_by_face_id'
GET_FACE_NAME_URL = 'get_name_by_face_id'
FACE_IMAGE = 'face.jpg'


def save_image(name, image):
    dir_file = os.path.dirname(__file__)
    path = f"{dir_file}/image/{name}"
    if not os.path.exists(path):
        os.mkdir(path)

    file_name = f"{name}_{time.time()}.jpg"

    with open(f"{path}/{file_name}", "wb") as fh:
        fh.write(image.decode('base64'))


def get_face_image():
    images_info = []
    url = create_url(GET_FACE_IMAGE_URL)
    response = requests.get(url)
    if response.status_code == 201 or response.status_code == 200:
        images_info = json.loads(response.content.decode())
        for i in range(len(images_info)):
            download_image(images_info[i]['ImageId'])
            image = cv2.imread(FACE_IMAGE)
            face_location, face_code = face_encode(image)
            if len(face_location) == 1:
                images_info[i]['Image'] = face_code[0]
    return images_info


def get_list_image_id_by_face_id(face_id):
    url = create_url(GET_IMAGE_ID_URL)
    response = requests.get(f'{url}/{face_id}')
    datas = []
    for data in json.loads(response.content.decode('UTF-8')):
        obj = {'im_id': data['ImageId'],
               'face_id': data['FaceId']}
        datas.append(obj)

    return datas


def get_name_by_face_id(face_id):
    name = 'Unknown'
    url = create_url(GET_FACE_NAME_URL)
    response = requests.get(f'{url}/{face_id}')
    data = json.loads(response.content.decode('UTF-8'))
    if face_id is not None and data and isinstance(data, list):
        name = data[0]['FaceName']
    return name


def download_image(image_id):
    url = create_url(DOWNLOAD_IMAGE_URL)
    response = requests.get(f"{url}{image_id}", stream=True)
    if response.status_code == 200:
        with open(FACE_IMAGE, 'wb') as f:
            for chunk in response:
                f.write(chunk)


def add_face(img_info):
    face_recognition_model.known_image_id.append(img_info['im_id'])
    face_recognition_model.known_face_id.append(img_info['face_id'])
    face_recognition_model.known_face_encodings.append(img_info['image'])


def encode_image_base64(image_base64):
    img_data = base64.b64decode(image_base64)
    image = Image.open(io.BytesIO(img_data))

    face_locations = face_recognition.face_locations(image)
    if len(face_locations) == 1:
        return face_recognition.face_encodings(image, face_locations)
    else:
        raise Exception("image must be  content only one face")


def training_face():
    face_recognition_model.known_face_encodings, face_recognition_model.known_face_id \
        , face_recognition_model.known_image_id = load_known_faces()
    faces_data = []
    images = get_face_image()
    for i in range(len(images)):
        if 'Image' in images[i]:
            obj = {'face_id': images[i]['FaceId'],
                   'im_id': images[i]['ImageId'],
                   'image': images[i]['Image']}
            faces_data.append(obj)
    for data in faces_data:
        add_face(data)

    save_known_faces(face_recognition_model.known_face_encodings, face_recognition_model.known_face_id,
                     face_recognition_model.known_image_id)
