import argparse
import time
import torch
import numpy as np

from pathlib import Path

from face_recognition_sdk import FaceRecognitionSDK
from face_recognition_sdk.utils.io_utils import read_yaml, read_image


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", "-p", help="path to image", type=str)
    parser.add_argument("--config", help="path to sdk config", type=str, default="face_recognition_sdk/config/config.yaml")
    args = parser.parse_args()

    config = read_yaml(args.config) if args.config else None

    sdk = FaceRecognitionSDK(config=config)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    img_path = Path(args.path).expanduser().resolve()
    img = read_image(img_path.as_posix())

    results = []
    for i in range(30):

        start_time = time.time()

        t = sdk.recognize_faces(img)

        if device == "cuda":
            torch.cuda.synchronize()

        time_taken = time.time() - start_time
        results.append(time_taken)

    results = np.asarray(results)
    print(f"MEAN RUN TIME: {results.mean()}")
    print(f"SUM TIME: {results.sum()}")
