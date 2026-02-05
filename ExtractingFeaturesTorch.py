import numpy as np 
import torch
from torchvision import transforms
# from PIL import Image

import torch.nn as nn


from vehicle_reid.load_model import load_model_from_opts

class ExtractingFeatures:
    def __init__(self):

        self.device = "cuda"
        self.model = load_model_from_opts("vehicle_reid/model/veri+vehixlex_editTrainPar1/opts.yaml", 
                                     ckpt="vehicle_reid/model/veri+vehixlex_editTrainPar1/net_39.pth", 
                                     remove_classifier=True)
        self.model.eval()
        self.model.to(self.device)

        self.data_transforms = transforms.Compose([
            transforms.Resize((224, 224), interpolation=3),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def fliplr(self, img):
        """flip images horizontally in a batch"""
        inv_idx = torch.arange(img.size(3) - 1, -1, -1).long()
        inv_idx = inv_idx.to(img.device)
        img_flip = img.index_select(3, inv_idx)
        return img_flip

    def extract_feature(self, model, X, device="cuda"):
        """Exract the embeddings of a single image tensor X"""
        # print("X")
        # print(X.shape)
        if len(X.shape) == 3:
            X = torch.unsqueeze(X, 0)
            # print("unsqueezed X")
            # print(X.shape)
        X = X.to(device)
        feature = model(X).reshape(-1)
        # print("extracted feature")


        X = self.fliplr(X)
        flipped_feature = model(X).reshape(-1)
        feature += flipped_feature

        fnorm = torch.norm(feature, p=2)
        return feature.div(fnorm)
    
    def get_feature(self, image, device="cuda"):

        image = [image]

        X_images = torch.stack(tuple(map(self.data_transforms, image))).to(device)

        features = [self.extract_feature(self.model, X_images)]
        features = torch.stack(features).detach().cpu()

        features_array = np.array(features)

        return features_array

