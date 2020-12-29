from flask import Blueprint, jsonify
from flask_api import status

from api.controler.config_controller import get_config_device
from api.services.camera_service import start_camera

config = Blueprint('config', __name__)


@config.route('/update')
def update_config():
    """
    api update config camera after change
    :return:
    """
    try:
        camera_data = get_config_device()

        if camera_data:
            start_camera(camera_data)

        response = {
            'status': 'success',
            'code': 200
        }
        return jsonify(response), status.HTTP_200_OK
    except Exception as ex:
        response = {
            'status': 'failed',
            'code': 400,
            'message': ex.__str__()
        }
        return jsonify(response), status.HTTP_400_BAD_REQUEST
