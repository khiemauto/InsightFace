import pickle
import os

import cv2

from .io_utils import read_image
from ..sdk import FaceRecognitionSDK


class FaceRecognitionSystem:

    """A simple class that demonstrates how Face SDK can be integrated into other systems."""

    def __init__(self, sdk_config: dict = None):

        self.user_id_to_name = {}
        self.sdk = FaceRecognitionSDK(sdk_config)

        self.descriptors_db_filename = "descriptors.index"
        self.id_to_user_filename = "id_to_username.pkl"

    def get_user_name(self, user_id: int) -> str:
        return self.user_id_to_name[user_id]

    def create_database_from_folders(self, root_path: str) -> None:
        """Create face database from hierarchy of folders.
        Each folder named as an individual and contains his/her photos.
        """

        try:
            for user_id, username in enumerate(os.listdir(root_path)):

                user_images_path = os.path.join(root_path, username)

                if not os.path.isdir(user_images_path):
                    continue

                self.user_id_to_name[user_id] = username

                # iterating over user photos
                for filename in os.listdir(user_images_path):
                    print(f"Adding {filename} from {user_images_path}")
                    photo_path = os.path.join(root_path, username, filename)
                    photo = read_image(photo_path)
                    self.sdk.add_photo_by_user_id(photo, user_id)
        except Exception:
            self.id_to_user_filename = {}
            self.sdk.reset_database()
            raise

    def save_database(self, folder_path: str) -> None:

        descriptors_path = os.path.join(folder_path, self.descriptors_db_filename)
        id_to_user_path = os.path.join(folder_path, self.id_to_user_filename)

        # save descriptors to userid index (sdk)
        self.sdk.save_database(descriptors_path)

        # save our own id to username mapping
        with open(id_to_user_path, "wb") as fp:
            pickle.dump(self.user_id_to_name, fp, protocol=pickle.HIGHEST_PROTOCOL)

    def load_database(self, folder_path: str) -> None:

        descriptors_path = os.path.join(folder_path, self.descriptors_db_filename)
        id_to_user_path = os.path.join(folder_path, self.id_to_user_filename)

        # load descriptors to userid index (sdk)
        self.sdk.load_database(descriptors_path)

        # load our own id to username mapping
        with open(id_to_user_path, "rb") as fp:
            self.user_id_to_name = pickle.load(fp)
