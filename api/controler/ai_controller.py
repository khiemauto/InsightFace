import cv2
import face_recognition

from api.services import face_service
from api.services.face_recognition_service import search_name_of_face, face_encode
from api.services.face_service import download_image
from model.face_recognition_model import FaceRecognitionModel, save_known_faces

face_recognition_model = FaceRecognitionModel.get_instance()
FACE_IMAGE = 'face.jpg'


def search_user_by_image(image):
    """ search face id by face image. If dont find anyone, return ''
    :param image: image content only one face
    :return: face id
    """
    face_id = None
    # Find all the faces and face encodings in the current frame of video
    face_locations = face_recognition.face_locations(image)
    if len(face_locations) == 1:
        face_encodings = face_recognition.face_encodings(image, face_locations)
        face_id = search_name_of_face(face_encodings[0])
    return face_id


def add_face(data):
    """
    add new face from win form
    :param data: list face data, every one item have im_id, face_id
    :return: None
    """
    for img_info in data:
        if img_info['im_id']:
            download_image(img_info['im_id'])
            image = cv2.imread(FACE_IMAGE)
            face_locations, face_encodings = face_encode(image)
            if len(face_locations) == 1:
                img_info['image'] = face_encodings[0]
                face_service.add_face(img_info)

    save_known_faces(face_recognition_model.known_face_encodings, face_recognition_model.known_face_id,
                     face_recognition_model.known_image_id)


def delete_face_by_user_id(user_id):
    """
    delete all face info of one user id
    :param user_id: user id
    :return: None
    """
    list_del = []
    for i in range(len(face_recognition_model.known_face_id)):
        if face_recognition_model.known_face_id[i] == user_id:
            list_del.append(i)

    face_recognition_model.known_face_id = [i for j, i in enumerate(face_recognition_model.known_face_id) if
                                            j not in list_del]
    face_recognition_model.known_face_encodings = [i for j, i in
                                                   enumerate(face_recognition_model.known_face_encodings) if
                                                   j not in list_del]
    face_recognition_model.known_image_id = [i for j, i in enumerate(face_recognition_model.known_image_id) if
                                             j not in list_del]


def delete_face_by_image_id(image_id):
    """
    delete face info of one image face
    :param image_id: image id
    :return: None
    """
    list_del = []
    for i in range(len(face_recognition_model.known_face_id)):
        if face_recognition_model.known_image_id[i] == image_id:
            list_del.append(i)

    face_recognition_model.known_face_id = [i for j, i in enumerate(face_recognition_model.known_face_id) if
                                            j not in list_del]
    face_recognition_model.known_face_encodings = [i for j, i in
                                                   enumerate(face_recognition_model.known_face_encodings) if
                                                   j not in list_del]
    face_recognition_model.known_image_id = [i for j, i in enumerate(face_recognition_model.known_image_id) if
                                             j not in list_del]
