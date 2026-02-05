import numpy as np 
import torch
from torchvision import transforms
from tensorrt_model import TensorRTModel
import cv2
# from PIL import Image

import torch.nn as nn


from vehicle_reid.load_model import load_model_from_opts

class ExtractingFeatures:
    def __init__(self):

        model_path = "vehicle_reid/model/veri+vehixlex_editTrainPar1/net_39_ubuntu.engine"
        self.batch_size = 1  # simulate streaming

        self.model = TensorRTModel(model_path)

        # ImageNet normalization values
        self.IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        self.IMAGE_TARGET_SIZE = (224, 224)

    def preprocess_image(self, img):
        img = cv2.resize(img, self.IMAGE_TARGET_SIZE , interpolation=cv2.INTER_CUBIC)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        img = (img - self.IMAGENET_MEAN) / self.IMAGENET_STD
        img = img.transpose(2, 0, 1)  # HWC -> CHW
        return img

    def fliplr_np(self, img_batch):
        """
        img_batch: numpy array of shape (B, C, H, W)
        Returns: horizontally flipped batch
        """
        return img_batch[:, :, :, ::-1]
    
    def l2_normalize(self, vec):
        norm = np.linalg.norm(vec) + 1e-12
        return vec / norm


    def extract_feature(self, X):
        """Exract the embeddings of a single transformed image X"""

        print("X shape:", X.shape)
        print("X dtype:", X.dtype)

        feature = self.model.infer(X).reshape(-1)

        print("feature shape:", feature.shape)
        print("feature dtype:", feature.dtype)

        X_flipped = self.fliplr_np(X)

        print("X_flipped shape:", X_flipped.shape)
        print("X_flipped dtype:", X_flipped.dtype)

        flipped_feature = self.model.infer(X_flipped).reshape(-1)

        print("flipped_feature shape:", flipped_feature.shape)
        print("flipped_feature dtype:", flipped_feature.dtype)

        feature += flipped_feature

        print("feature after flipping shape:", feature.shape)
        print("feature after flipping dtype:", feature.dtype)

        fnorm = self.l2_normalize(feature)

        print("fnorm shape:", fnorm.shape)
        print("fnorm dtype:", fnorm.dtype)

        return fnorm
    
    def get_feature(self, image):
        """Extract features from a single image (Currently)."""
        # image = Opencv array in BGR format
        img = self.preprocess_image(image)

        # img preprocess shape: (3, 224, 224)
        # img preprocess dtype: float32

        feats = self.extract_feature([img])

        print("feats[0] shape:", feats[0].shape)
        print("feats[0] dtype:", feats[0].dtype)

        return np.array(feats[0])
    
