import numpy as np

from typing import Tuple

from .align_faces import warp_and_crop_face


def align_and_crop_face(image: np.ndarray, facial5points: np.ndarray, size: Tuple = (96, 112)) -> np.ndarray:
    """Crop and align face on the image given keypoints.
    Args:
        image: numpy image in BGR format.
        facial5points: array of image keypoints.
        size: crop size (width, height). Either (96, 112) or (112, 112)
    """

    facial5points = np.reshape(facial5points, (2, 5))
    dst_img = warp_and_crop_face(image, facial5points, crop_size=size)
    return dst_img
