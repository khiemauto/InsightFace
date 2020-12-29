from model.camera_model import CameraModel

cameras = []
device_ids = []


def stop_camera():
    for camera in cameras:
        camera.run_thread = False


def start_camera(camera_data):
    stop_camera()
    cameras.clear()
    device_ids.clear()
    for data in camera_data:
        camera = CameraModel(data)
        cameras.append(camera)
        device_ids.append(str(data['DeviceId']))


def get_camera(device_id):
    if str(device_id) in device_ids:
        index = device_ids.index(device_id)
        return cameras[index]
    else:
        return None
