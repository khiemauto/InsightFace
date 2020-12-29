import argparse
import cv2
import os

from tqdm.auto import tqdm
from pathlib import Path

from pygame import mixer

from face_recognition_sdk.utils.database import FaceRecognitionSystem
from face_recognition_sdk.utils.io_utils import read_image, save_image
from face_recognition_sdk.utils.draw_utils import draw_boxes, draw_landmarks

from gtts import gTTS


def generate_greeting(username):

    mp3_fp = "greeting.mp3"
    text = f"Hello, {username}!"
    tts = gTTS(text, lang="en")
    # tts.write_to_fp(mp3_fp)
    tts.save(mp3_fp)

    return mp3_fp


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--folders_path",
        "-fp",
        help="path to save folders with images",
        default=None,
    )
    parser.add_argument(
        "--db_folder_path",
        "-dbp",
        help="path to save database",
        default="../employees/database",
    )

    args = parser.parse_args()

    system = FaceRecognitionSystem()

    folders_path = args.folders_path
    db_folder_path = args.db_folder_path

    # create, save and load database initialized from folders containing user photos
    if folders_path is not None:
        system.create_database_from_folders(folders_path)
        system.save_database(db_folder_path)

    system.load_database(db_folder_path)

    cap = cv2.VideoCapture(0)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    cv2.namedWindow("Webcam", cv2.WINDOW_NORMAL)

    mixer.init()

    try:
        while True:
            ret, frame = cap.read()

            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            bboxes, landmarks, user_ids, similarities = system.sdk.recognize_faces(img)
            names = [system.get_user_name(uid) for uid in user_ids]

            draw_boxes(img, bboxes, name_tags=names, similarities=similarities)
            draw_landmarks(img, landmarks)

            for name in names:
                greeting = generate_greeting(name)
                mixer.music.load(greeting)
                mixer.music.play()

            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

            cv2.imshow("Webcam", img)

            keyPressed = cv2.waitKey(1)

            if keyPressed == 27 or keyPressed == 1048603:
                break  # esc to quit

    finally:
        cv2.destroyAllWindows()
        cap.release()
        os.remove("greeting.mp3")