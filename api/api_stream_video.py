from flask import Blueprint, Response, jsonify, render_template

from api.controler.config_controller import get_config_device
from api.services.camera_service import get_camera

stream = Blueprint('stream', __name__)
get_in = [False]
get_out = [False]


def gen_in(camera):
    """
    generator frame video
    :param camera: camera
    :return:
    """
    while get_in[0]:
        print(f"{camera.device_id}")
        try:
            frame = camera.get_video_stream()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
        except Exception:
            continue


def gen_out(camera):
    """
    generator frame video
    :param camera: camera
    :return:
    """
    while get_out[0]:
        print(f"{camera.device_id}")
        try:
            frame = camera.get_video_stream()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
        except Exception:
            continue


@stream.route('/in/video_feed/<device_id>')
def video_feed_in(device_id):
    """
    api get video of camera
    :param device_id: device id
    :return:
    """
    get_in[0] = False
    if device_id:
        camera = get_camera(device_id)
        if camera:
            get_in[0] = True
            return Response(gen_in(camera),
                            mimetype='multipart/x-mixed-replace; boundary=frame')


@stream.route('/out/video_feed/<device_id>')
def video_feed_out(device_id):
    """
    api get video of camera
    :param device_id: device id
    :return:
    """
    get_out[0] = False
    if device_id:
        camera = get_camera(device_id)
        if camera:
            get_out[0] = True
            return Response(gen_out(camera),
                            mimetype='multipart/x-mixed-replace; boundary=frame')


@stream.route('/in')
def index_in():
    return render_template('index_in.html')


@stream.route('/out')
def index_out():
    return render_template('index_out.html')


@stream.route('/get_all_camera')
def get_all_camera():
    """
    get all camera
    :return: list camera
    """
    data = get_config_device()
    response = jsonify(data=data,
                       code=200)
    response.headers['Access-Control-Allow-Origin'] = '*'
    # return JSON
    return response
