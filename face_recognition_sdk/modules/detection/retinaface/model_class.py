import torch
import numpy as np
import yaml
import cv2

from torch import Tensor
from pathlib import Path
from typing import Tuple

from ..base_detector import BaseFaceDetector
from .dependencies.retinaface import RetinaFace as ModelClass
from face_recognition_sdk.utils.load_utils import load_model
from .dependencies.utils import decode, decode_landm, py_cpu_nms
from .dependencies.prior_box import PriorBox

model_urls = {
    "res50": "https://face-demo.indatalabs.com/weights/Resnet50_Final.pth",
    "mnet1": "https://face-demo.indatalabs.com/weights/mobilenet0.25_Final.pth",
}


class RetinaFace(BaseFaceDetector):
    def __init__(self, config: dict):
        """
        Args:
            config: detector config for configuration of model from outside
        """
        super().__init__(config)
        backbone = config["architecture"]
        path_to_model_config = Path(Path(__file__).parent, "config.yaml").as_posix()
        with open(path_to_model_config, "r") as fp:
            self.model_config = yaml.load(fp, Loader=yaml.FullLoader)

        if backbone not in self.model_config.keys():
            raise ValueError(f"Unsupported backbone: {backbone}!")

        self.model_config = self.model_config[backbone]
        self.model = ModelClass(self.model_config, phase="test")
        self.device = torch.device(self.config["device"])
        self.model = load_model(self.model, model_urls[backbone], True if self.config["device"] == "cpu" else False)
        self.model.eval()
        self.model = self.model.to(self.device)
        self.model_input_shape = None
        self.resize_scale = None

    def _preprocess(self, image: np.ndarray) -> np.ndarray:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        img = np.float32(image)
        target_size = self.config["image_size"]
        max_size = 2150
        im_shape = img.shape
        im_size_min = np.min(im_shape[:2])
        im_size_max = np.max(im_shape[:2])
        resize = float(target_size) / float(im_size_min)

        # prevent bigger axis from being more than max_size:
        if np.round(resize * im_size_max) > max_size:
            resize = float(max_size) / float(im_size_max)
        if self.config.get("origin_size", False):
            resize = 1

        self.resize_scale = resize
        if resize != 1:
            img = cv2.resize(image, None, None, fx=resize, fy=resize, interpolation=cv2.INTER_LINEAR)

        img = img.astype(np.float32)
        img -= np.asarray((104, 117, 123), dtype=np.float32)
        return img

    def _predict_raw(self, image: np.ndarray) -> Tuple[Tensor, Tensor, Tensor]:
        img = image.transpose((2, 0, 1))
        img = torch.from_numpy(img).unsqueeze(0)
        img = img.to(self.device)
        pred = self.model(img)
        # loc = loc.detach().cpu().numpy()
        # conf = conf.detach().cpu().numpy()
        # landms = landms.detach().cpu().numpy()
        return pred

    def _postprocess(self, raw_prediction: Tuple[Tensor, Tensor, Tensor]) -> Tuple[np.ndarray, np.ndarray]:
        loc, conf, landms = raw_prediction
        img_h, img_w = self.model_input_shape[:2]
        scale = torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float)
        scale = scale.to(self.device)

        priorbox = PriorBox(self.model_config, image_size=(img_h, img_w))
        priors = priorbox.forward()
        priors = priors.to(self.device)
        prior_data = priors.data
        boxes = decode(loc.data.squeeze(0), prior_data, self.model_config["variance"])
        boxes = boxes * scale / self.resize_scale
        boxes = boxes.cpu().numpy()

        scores = conf.squeeze(0).data.cpu().numpy()[:, 1]

        landms = decode_landm(landms.data.squeeze(0), prior_data, self.model_config["variance"])
        scale1 = torch.tensor([img_w, img_h, img_w, img_h, img_w, img_h, img_w, img_h, img_w, img_h], dtype=torch.float)
        scale1 = scale1.to(self.device)
        landms = landms * scale1 / self.resize_scale
        landms = landms.cpu().numpy()

        # ignore low scores
        inds = np.where(scores > self.config["conf_threshold"])[0]
        boxes = boxes[inds]
        landms = landms[inds]
        scores = scores[inds]

        # keep top-K before NMS
        order = scores.argsort()[::-1]
        # order = scores.argsort()[::-1][:args.top_k]
        boxes = boxes[order]
        landms = landms[order]
        scores = scores[order]

        # do NMS
        dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
        keep = py_cpu_nms(dets, self.config["nms_threshold"])
        # keep = nms(dets, args.nms_threshold,force_cpu=args.cpu)
        dets = dets[keep, :]
        landms = landms[keep]

        return dets, landms

    def predict(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        image = self._preprocess(image)
        self.model_input_shape = image.shape
        raw_pred = self._predict_raw(image)
        bboxes, landms = self._postprocess(raw_pred)

        converted_landmarks = []
        # convert to our landmark format (2,5)
        for landmarks_set in landms:
            x_landmarks = []
            y_landmarks = []
            for i, lm in enumerate(landmarks_set):
                if i % 2 == 0:
                    x_landmarks.append(lm)
                else:
                    y_landmarks.append(lm)
            converted_landmarks.append(x_landmarks + y_landmarks)

        landmarks = np.array(converted_landmarks)

        return bboxes, landmarks

    def _get_raw_model(self):
        return self.model
