import argparse
import cv2

from pathlib import Path

from face_recognition_sdk.utils.database import FaceRecognitionSystem
from face_recognition_sdk.utils.io_utils import read_yaml, read_image, save_image
from face_recognition_sdk.utils.draw_utils import draw_boxes, draw_landmarks


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", "-p", help="path to image", type=str)
    parser.add_argument(
        "--config", help="path to sdk config", type=str, default="face_recognition_sdk/config/config.yaml"
    )
    parser.add_argument("--result_path", "-r", help="path to save processed image", default="demo/results/")
    parser.add_argument(
        "--folders_path",
        "-fp",
        help="path to save folders with images",
        default="/home/d_barysevich/FaceRecognition/employees/images",
    )
    parser.add_argument(
        "--db_folder_path",
        "-dbp",
        help="path to save database",
        default="/home/d_barysevich/FaceRecognition/employees/database",
    )
    args = parser.parse_args()

    sdk_config = read_yaml(args.config)

    system = FaceRecognitionSystem(sdk_config)

    folders_path = args.folders_path
    db_folder_path = args.db_folder_path

    # create, save and load database initialized from folders containing user photos
    if folders_path is not None:
        system.create_database_from_folders(folders_path)
        system.save_database(db_folder_path)

    system.load_database(db_folder_path)

    # search user on query image in our database
    img_path = Path(args.path).expanduser().resolve()
    img = read_image(img_path.as_posix())

    bboxes, landmarks, user_ids, similarities = system.sdk.recognize_faces(img)
    names = [system.get_user_name(uid) for uid in user_ids]

    draw_boxes(img, bboxes, name_tags=names, similarities=similarities)
    draw_landmarks(img, landmarks)

    result_path = Path(args.result_path).expanduser().resolve()
    if not result_path.exists():
        result_path.mkdir(parents=True)

    path_to_save = Path(result_path, img_path.name)

    save_image(img, path_to_save.as_posix())
