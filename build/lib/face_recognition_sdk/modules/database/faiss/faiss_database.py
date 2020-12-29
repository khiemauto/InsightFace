from typing import Union, Tuple, List
import faiss
import numpy as np

from ..base_storage import BaseFaceStorage


class FaissFaceStorage(BaseFaceStorage):
    """
    Face storage using FAISS backend for similarity search.
    """

    def __init__(self, config: dict):

        self.descriptor_size = config["descriptor_size"]
        self.reset()

    def load(self, path: str) -> None:
        """Load database."""
        self.index = faiss.read_index(path)

    def save(self, path: str) -> None:
        """Save database."""
        faiss.write_index(self.index, path)

    def reset(self) -> None:
        self.index = faiss.IndexIDMap(faiss.IndexFlatIP(self.descriptor_size))

    def find(self, descriptor: np.ndarray, top_k: int) -> List[Tuple[int, int, np.ndarray]]:
        """Add descriptor with specified user_id."""
        if descriptor.ndim == 1:
            descriptor = np.expand_dims(descriptor, 0)
        distances, indicies = self.index.search(descriptor, top_k)
        return indicies[0], distances[0]

    def add_descriptor(self, descriptor: np.ndarray, user_id: int) -> None:
        """Add descriptor with specified user_id.

        Args:
            descriptor: np.array [1,D] of type float32
            user_id: id of the user
        """
        if descriptor.ndim == 1:
            descriptor = np.expand_dims(descriptor, 0)
        self.index.add_with_ids(descriptor, np.array([user_id]))

    def remove_descriptor(self, descriptor_id: int) -> None:
        """Updates user id list of descriptor ids."""
        raise NotImplementedError

    def remove_user(self, user_id: int) -> None:
        """Removes all user descriptors from the database."""
        self.index.remove_ids(np.array([user_id]))
