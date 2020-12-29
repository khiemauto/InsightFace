import argparse
import cv2

from tqdm.auto import tqdm
from pathlib import Path

from face_recognition_sdk import FaceRecognitionSDK
from face_recognition_sdk.utils.io_utils import read_yaml
from face_recognition_sdk.utils.draw_utils import draw_boxes, draw_landmarks

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", "-p", help="path to video", type=str)
    parser.add_argument(
        "--config", help="path to sdk config", type=str, default="face_recognition_sdk/config/config.yaml"
    )
    parser.add_argument("--result_path", "-r", help="path to save processed video", default="demo/results")
    args = parser.parse_args()

    config = read_yaml(args.config)

    sdk = FaceRecognitionSDK(config=config)

    video_path = Path(args.path).expanduser().resolve()

    result_path = Path(args.result_path).expanduser().resolve()
    if not result_path.exists():
        result_path.mkdir(parents=True)

    path_to_save = Path(result_path, video_path.name).as_posix()

    cap = cv2.VideoCapture(video_path.as_posix())

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video_writer = cv2.VideoWriter(path_to_save, fourcc, 24.0, (width, height))

    video_len = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    for _ in tqdm(range(video_len)):
        ret, frame = cap.read()
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        boxes, landms = sdk.detect_faces(img)

        draw_boxes(img, boxes)
        draw_landmarks(img, landms)

        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        video_writer.write(img)

    cap.release()
    video_writer.release()
