from typing import List
import numpy as np
import torch
from torchvision import transforms

from . import nets

from ..base_embedder import BaseFaceEmbedder
from torch2trt import torch2trt, TRTModule
from pathlib import Path

class InsightFaceEmbedder(BaseFaceEmbedder):

    """Implements inference of face recognition nets from InsightFace project."""

    def __init__(self, config: dict):

        self.device = config["device"]
        architecture = config["architecture"]

        trt_path = f"{architecture}.trt"
        trt_path = Path(Path(__file__).parent, trt_path)

        if not trt_path.exists():
            if architecture == "iresnet34":
                self.embedder = nets.iresnet34(pretrained=True)
            elif architecture == "iresnet50":
                self.embedder = nets.iresnet50(pretrained=True)
            elif architecture == "iresnet100":
                self.embedder = nets.iresnet100(pretrained=True)
            else:
                raise ValueError(f"Unsupported network architecture: {architecture}")
            self.embedder.eval()
            self.embedder.to(self.device)
            print(f"[INFO] Building {architecture} tensorrt")
            x = torch.ones((1,3,112,112)).to(self.device)
            self.embedder = torch2trt(self.embedder, [x], max_batch_size=32, fp16_mode=False)
            torch.save(self.embedder.state_dict(), trt_path)
        else:
            print(f"[INFO] Loading {architecture} tensorrt")
            self.embedder = TRTModule()
            self.embedder.load_state_dict(torch.load(trt_path))

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

    #Batch run
    def _preprocess_batch(self, faces: List[np.ndarray]) -> List[torch.Tensor]:
        face_tensors = []
        for face in faces:
            face_tensors.append(self.preprocess(face))
        return face_tensors

    def _predict_raw_batch(self, faces: List[torch.Tensor]) -> torch.Tensor:
        with torch.no_grad():
            faces = torch.stack(faces).to(self.device)
            features = self.embedder(faces)
        return features

    def _postprocess_batch(self, raw_predictions: torch.Tensor) -> np.ndarray:
        raw_predictions = raw_predictions.cpu().numpy()
        descriptors = raw_predictions / np.linalg.norm(raw_predictions, axis=1)[:,None]
        return descriptors
    
    def run_batch(self, images: List[np.ndarray]) -> List[np.ndarray]:
        return self._postprocess_batch(self._predict_raw_batch(self._preprocess_batch(images)))