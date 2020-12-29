import argparse
import cv2

from pathlib import Path

from face_recognition_sdk import FaceRecognitionSDK
from face_recognition_sdk.utils.io_utils import read_yaml
from face_recognition_sdk.utils.draw_utils import draw_boxes, draw_landmarks


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", "-p", help="path to image", type=str)
    parser.add_argument(
        "--config", help="path to sdk config", type=str, default="face_recognition_sdk/config/config.yaml"
    )
    parser.add_argument("--result_path", "-r", help="path to save processed image", default="demo/results")
    args = parser.parse_args()

    config = read_yaml(args.config)

    sdk = FaceRecognitionSDK(config=config)

    img_path = Path(args.path).expanduser().resolve()
    img = cv2.imread(img_path.as_posix())
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    boxes, landms = sdk.detect_faces(img)

    draw_boxes(img, boxes)
    draw_landmarks(img, landms)

    result_path = Path(args.result_path).expanduser().resolve()
    if not result_path.exists():
        result_path.mkdir(parents=True)

    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    path_to_save = Path(result_path, img_path.name)
    cv2.imwrite(path_to_save.as_posix(), img)
