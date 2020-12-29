import numpy as np
import torch
from torchvision import transforms

from . import nets

from ..base_embedder import BaseFaceEmbedder


class InsightFaceEmbedder(BaseFaceEmbedder):

    """Implements inference of face recognition nets from InsightFace project."""

    def __init__(self, config: dict):

        self.device = config["device"]
        architecture = config["architecture"]

        if architecture == "iresnet34":
            self.embedder = nets.iresnet34(pretrained=True)
        elif architecture == "iresnet50":
            self.embedder = nets.iresnet50(pretrained=True)
        elif architecture == "iresnet100":
            self.embedder = nets.iresnet100(pretrained=True)
        else:
            raise ValueError(f"Unsupported network architecture: {architecture}")

        self.embedder.to(self.device)
        self.embedder.eval()

        mean = [0.5] * 3
        std = [0.5 * 256 / 255] * 3

        self.preprocess = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])

    def _preprocess(self, face: np.ndarray) -> np.ndarray:
        face_tensor = self.preprocess(face).unsqueeze(0).to(self.device)
        return face_tensor

    def _predict_raw(self, face: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            features = self.embedder(face)
        return features

    def _postprocess(self, raw_prediction: np.ndarray) -> np.ndarray:
        descriptor = raw_prediction[0].cpu().numpy()
        descriptor = descriptor / np.linalg.norm(descriptor)
        return descriptor

    def _get_raw_model(self):
        return self.embedder
