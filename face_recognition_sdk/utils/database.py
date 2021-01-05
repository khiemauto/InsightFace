import pickle
import os

import cv2

from .io_utils import read_image
from ..sdk import FaceRecognitionSDK


class FaceRecognitionSystem:

    """A simple class that demonstrates how Face SDK can be integrated into other systems."""

    def __init__(self, sdk_config: dict = None):
        self.photoid_to_username_photopath = {}
        self.sdk = FaceRecognitionSDK(sdk_config)

        self.descriptors_db_filename = "descriptors.index"
        self.photoid_to_username_photopath_filename = "id_to_username.pkl"

    def get_user_name(self, photo_id: int) -> str:
        return self.photoid_to_username_photopath[photo_id][0]
    
    def get_photo_path(self, photo_id: int) -> str:
        return self.photoid_to_username_photopath[photo_id][1]

    def create_database_from_folders(self, root_path: str) -> None:
        """Create face database from hierarchy of folders.
        Each folder named as an individual and contains his/her photos.
        """

        try:
            photo_id = 0
            for username in os.listdir(root_path):

                user_images_path = os.path.join(root_path, username)

                if not os.path.isdir(user_images_path):
                    continue

                # iterating over user photos
                for filename in os.listdir(user_images_path):
                    print(f"Adding {filename} from {user_images_path}")
                    photo_path = os.path.join(root_path, username, filename)
                    photo = read_image(photo_path)
                    self.sdk.add_photo_by_photo_id(photo, photo_id)
                    self.photoid_to_username_photopath[photo_id] = [username, photo_path]
                    photo_id += 1
        except Exception:
            self.photoid_to_username_photopath = {}
            self.sdk.reset_database()
            raise

    def save_database(self, folder_path: str) -> None:

        descriptors_path = os.path.join(folder_path, self.descriptors_db_filename)
        id_to_user_path = os.path.join(folder_path, self.photoid_to_username_photopath_filename)

        # save descriptors to userid index (sdk)
        self.sdk.save_database(descriptors_path)

        # save our own id to username mapping
        with open(id_to_user_path, "wb") as fp:
            pickle.dump(self.photoid_to_username_photopath, fp, protocol=pickle.HIGHEST_PROTOCOL)

    def load_database(self, folder_path: str) -> None:

        descriptors_path = os.path.join(folder_path, self.descriptors_db_filename)
        id_to_user_path = os.path.join(folder_path, self.photoid_to_username_photopath_filename)

        # load descriptors to userid index (sdk)
        self.sdk.load_database(descriptors_path)

        # load our own id to username mapping
        with open(id_to_user_path, "rb") as fp:
            self.photoid_to_username_photopath = pickle.load(fp)