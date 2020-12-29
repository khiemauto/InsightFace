import datetime
import json
import sys
import time

import cv2
import face_recognition
import numpy as np
import requests

from core.constant import Constant
from core.helper import create_url
from model.face_recognition_model import FaceRecognitionModel, save_known_faces

face_recognition_model = FaceRecognitionModel.get_instance()
constant = Constant.get_instance()

FACE_UPLOAD = 'face_upload'
FACE_IMAGE = 'face.jpg'


def face_encode(face):
    face_locations = face_recognition.face_locations(face)
    face_encodings = face_recognition.face_encodings(face, face_locations)
    return face_locations, face_encodings


def search_name_of_face(face_encoding):
    face_id = None

    if face_recognition_model.known_face_encodings:
        matches = face_recognition.compare_faces(face_recognition_model.known_face_encodings, face_encoding,
                                                 tolerance=0.4)
        # Or instead, use the known face with the smallest distance to the new face
        face_distances = face_recognition.face_distance(face_recognition_model.known_face_encodings, face_encoding)
        best_match_index = np.argmin(face_distances)
        if matches[best_match_index]:
            face_id = face_recognition_model.known_face_id[best_match_index]

    return face_id


def save_face(face_image, face_id):
    """
    save face is caught is show
    :param face_image: face is caught
    :param face_id: name of face
    :return:
    """

    exist = False
    for index, obj in enumerate(face_recognition_model.faces_caught):
        if obj["name"] == face_id:
            exist = True
            time_now = time.time()
            old_time = float(obj["time"])
            if time_now - old_time > constant.TIME_CAUGHT_FACE:
                # update object time
                obj["time"] = time_now.__str__()
                face_recognition_model.faces_caught[index] = obj
            break

    if not exist:
        if face_id == face_recognition_model.unknown:
            face_id = f"{face_id}_{time.time()}"
        obj = {
            "time": time.time(),
            "name": face_id
        }
        face_recognition_model.faces_caught.append(obj)

        # append new face caught
        face_recognition_model.known_face_encodings.append(face_recognition.face_encodings(face_image))
        face_recognition_model.known_face_id.append(face_id)

        # backup training model
        save_known_faces(face_recognition_model.known_face_encodings, face_recognition_model.known_face_id,
                         face_recognition_model.known_face_id)


def recognition_face(frame):
    try:
        # Resize frame of video to 1/2 size for faster face recognition processing
        small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)

        # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
        rgb_small_frame = small_frame[:, :, ::-1]

        # Only process every other frame of video to save time

        # Find all the faces and face encodings in the current frame of video
        face_locations, face_encodings = face_encode(rgb_small_frame)

        track_time = datetime.datetime.now()

        face_ids = []
        for face_encoding in face_encodings:
            face_id = search_name_of_face(face_encoding)
            face_ids.append(face_id)

        # Display the results
        for (top, right, bottom, left) in face_locations:
            # Scale back up face locations since the frame we detected in was scaled to 1/2 size
            top *= 2
            right *= 2
            bottom *= 2
            left *= 2

            # Draw a box around the face
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 84, 255), 2)
        return frame, face_ids, track_time, face_encodings, face_locations
    except Exception as ex:
        print(__file__)
        print("Error on line {}".format(sys.exc_info()[-1].tb_lineno))
        print(ex)
        return None, None, None


def send_face_rec(_face_id, device_id, track_time, frame):
    face_id = None
    url = create_url(FACE_UPLOAD)
    data = {'faceId': _face_id,
            'eventId': 1,
            'deviceId': device_id,
            'RecordTime': track_time.strftime("%Y-%m-%d %H:%M:%S")}

    cv2.imwrite(FACE_IMAGE, frame)

    file = {'file': open(FACE_IMAGE, 'rb')}
    r = requests.post(url, files=file, params=data)
    # r = requests.post(url, files=file, header=data)
    print(r.status_code)
    print(f"send face id: {_face_id}")
    print(f"send device id: {device_id}")
    if r.status_code == 201:
        face_id = json.loads(r.content)['Message']
    return face_id
