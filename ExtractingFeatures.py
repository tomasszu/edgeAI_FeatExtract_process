import numpy as np 
from tensorrt_model import TensorRTModel
import cv2
# from PIL import Image

class ExtractingFeatures:
    def __init__(self):

        model_path = "model/veri+vehixlex_editTrainPar1/net_39_opt_jetson_built.engine"
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

        # X shape: (3, 224, 224)
        # X dtype: float32

        # If X is already a single image NumPy array of shape (3, 224, 224), you should wrap it in a list first before stacking. Otherwise np.stack treats the (3, 224, 224) array as three “items” along axis 0, giving a shape (3, 3, 224, 224), which explains your reshape error.

        # Ensure X is a list of images
        if isinstance(X, np.ndarray):
            X = [X]  # wrap single image in a list


        #Tā inffer funkcija tomēr sagaida ar batch dimension (kas ir labi - nākotnē implementēt)
        # Tāpēc uztaisam numpy stack lai viņš var noteikt .shape un uzzināt ka batch ir 1
        X = np.stack(X, axis=0)  # now X.shape = (batch, 3, 224, 224)

        # Savukārt reshape nevajag (noņēmu) - flatteno feature vektoru
        # Piemēram, (2, 2048) -> (4096,)

        feature = self.model.infer(X)[0]  # Get the first (and only) batch

        # feature shape: (256,)
        # feature dtype: float32

        X_flipped = self.fliplr_np(X)

        # X_flipped shape: (1, 3, 224, 224)
        # X_flipped dtype: float32

        flipped_feature = self.model.infer(X_flipped)[0]  # Get the first (and only) batch

        #flipped_feature shape: (256,)
        #flipped_feature dtype: float32

        feature += flipped_feature

        fnorm = self.l2_normalize(feature)

        # fnorm shape: (256,)
        # fnorm dtype: float32

        return fnorm
    
    def get_feature(self, image):
        """Extract features from a single image (Currently)."""
        # image = Opencv array in BGR format
        img = self.preprocess_image(image)

        # img preprocess shape: (3, 224, 224)
        # img preprocess dtype: float32
        

        feats = self.extract_feature(img)

        return feats
    
