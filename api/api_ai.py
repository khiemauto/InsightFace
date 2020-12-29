import json
import os

import cv2
from flask import jsonify, Blueprint, request
from flask_api import status

from api.controler import ai_controller
from api.services.face_service import get_list_image_id_by_face_id

ai = Blueprint('ai', __name__)
ALLOWED_EXTENSIONS = {'png', 'jpg'}
FACE_IMAGE = 'face.jpg'


def allowed_file(filename):
    """
    check type of file
    :param filename: file name
    :return: boolean
    """
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@ai.route('/get-face-id-by-face-image', methods=['POST'])
def search_face_by_image():
    """
    api search face by image
    """
    try:
        # check have file send
        if 'file' not in request.files:
            raise Exception("No file part")

        file = request.files['file']

        # check file select
        if file.filename == '':
            raise Exception("No selected file")

        # check file type
        if file and allowed_file(file.filename):
            file.save(FACE_IMAGE)

        # encode face image
        image = cv2.imread(FACE_IMAGE)

        # search face id
        face_id = ai_controller.search_user_by_image(image)
        response = {
            'status': 'success',
            'code': 200,
            'data': {
                'face_id': face_id
            }
        }
        return jsonify(response)
    except Exception as ex:
        response = {
            'status': 'fail',
            'code': 400,
            'message': ex.__str__()
        }
        return jsonify(response), status.HTTP_400_BAD_REQUEST


@ai.route('/face/add/<face_id>')
def add_new_face(face_id=None):
    """
    api add new face
    :param face_id: fae id
    :return:
    """
    try:
        content = None
        # get list image id by face id
        if face_id:
            content = get_list_image_id_by_face_id(face_id)

        # add face
        ai_controller.add_face(content)

        response = {
            'status': 'success',
            'code': 200,
        }
        return jsonify(response), status.HTTP_200_OK
    except Exception as ex:
        response = {
            'status': 'failed',
            'code': 400,
            'message': ex.__str__()
        }
        return jsonify(response), status.HTTP_400_BAD_REQUEST


@ai.route('/face/delete-by-face-id', methods=['PUT'])
def delete_face_by_user_id():
    """
    api delete face by face id
    :return:
    """
    try:
        data = request.data.decode('utf8').replace("'", '"')
        content = json.loads(data)
        user_id = content['user_id']

        ai_controller.delete_face_by_user_id(user_id)

        response = {
            'status': 'success',
            'code': 200,
        }
        return jsonify(response)
    except Exception as ex:
        response = {
            'status': 'failed',
            'code': 400,
            'message': ex.__str__()
        }
        return jsonify(response), status.HTTP_400_BAD_REQUEST


@ai.route('/face/delete-by-image-id', methods=['PUT'])
def delete_face_by_image_id():
    """
    api delete face by image id
    :return:
    """
    try:
        data = request.data.decode('utf8').replace("'", '"')
        content = json.loads(data)
        image_id = content['image_id']

        ai_controller.delete_face_by_image_id(image_id)

        response = {
            'status': 'success',
            'code': 200,
        }
        return jsonify(response)
    except Exception as ex:
        response = {
            'status': 'failed',
            'code': 400,
            'message': ex.__str__()
        }
        return jsonify(response), status.HTTP_400_BAD_REQUEST
