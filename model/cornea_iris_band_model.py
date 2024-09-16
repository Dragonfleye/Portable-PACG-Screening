from mmseg.apis import init_model, inference_model
import cv2
import numpy as np
from PIL import Image

IMAGE_SIZE = 1024
config_path = '/home/ubuntu/projects/glaucoma/config/cornea_iris_band.py'
checkpoint_path = '/home/ubuntu/projects/glaucoma/pth/cornea_iris_band.pth'


def cornea_iris_band_infer(img, mask):
    # inputs = np.array(img)[..., [2, 1, 0]]
    # inputs = np.transpose(img, (2, 0, 1))
    model = init_model(config_path, checkpoint_path, device='cuda:0')
    result = inference_model(model, img)
    segment = result.pred_sem_seg.data.cpu().numpy()
    segment = segment[0].astype(np.uint8).squeeze()
    segment = segment * mask
    return segment