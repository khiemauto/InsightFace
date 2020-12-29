import torch
import numpy as np
import albumentations as A
import json

from albumentations.pytorch import ToTensorV2

from ..base_attribute_classifier import BaseAttributeClassifier
from face_recognition_sdk.utils.load_utils import get_file_from_url


model_urls = {
    "res18": "https://face-demo.indatalabs.com/weights/attr_resnet18_jit_best.pt",
    "mbnet2": "https://face-demo.indatalabs.com/weights/attr_mbnet2_jit_best.pt",
}


class AttributeClassifierV1(BaseAttributeClassifier):
    """
    Implements inference for attribute classifier
    """

    def __init__(self, config):
        super().__init__(config)
        arch = config["architecture"]

        model_path = get_file_from_url(model_urls[arch], progress=True, unzip=False)
        self.model = torch.jit.load(model_path)
        self.device = torch.device(self.config["device"])
        self.model.to(self.device)

        hparams = json.loads(self.model.hparams)
        self.preprocess = A.from_dict(hparams["preprocess"])

        self.threshold = self.config["decision_threshold"]
        self.categories = hparams["categories"]

    def _preprocess(self, image: np.ndarray) -> torch.Tensor:
        preprocessed = self.preprocess(image=image)
        img = preprocessed["image"].to(self.device)
        return img

    def _predict_raw(self, image: torch.Tensor) -> torch.Tensor:
        image = image.unsqueeze(0)
        return self.model(image)[0]

    def _postprocess(self, raw_prediction: torch.Tensor) -> dict:
        prediction = torch.sigmoid(raw_prediction)
        prediction[prediction >= self.threshold] = 1
        prediction[prediction < self.threshold] = 0
        prediction = prediction.detach().cpu().numpy()
        res = {}
        for i, cat in enumerate(self.categories):
            res[cat] = int(prediction[i])
        return res

    def predict(self, image: np.ndarray) -> dict:
        image = self._preprocess(image)
        raw_pred = self._predict_raw(image)
        res = self._postprocess(raw_pred)
        return res
