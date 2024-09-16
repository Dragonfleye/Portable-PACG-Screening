from mmseg.apis import init_model, inference_model
import cv2
import numpy as np
from PIL import Image
BORDER = 1024
IMAGE_SIZE = 2048
FOV_DIAMETER = 1948
config_path = '/home/ubuntu/projects/glaucoma/config/limbus.py'
checkpoint_path = '/home/ubuntu/projects/glaucoma/pth/limbus.pth'


def limbus_infer(img_path):
    model = init_model(config_path, checkpoint_path, device='cuda:0')
    result = inference_model(model, img_path)
    segment = result.pred_sem_seg.data.cpu().numpy()
    segment = segment[0].astype(np.uint8).squeeze()
    contours, hierarchy = cv2.findContours(segment, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnt = contours[0]
    if len(contours) == 0:
        print("未找到角巩膜缘")
    if len(contours) > 1:
        max_large_index = np.argmax(np.array([len(cnt) for cnt in contours]))
        cnt = contours[max_large_index]
    rect = cv2.minAreaRect(cnt)
    diameter = max(rect[1])

    x, y, w, h = cv2.boundingRect(cnt)
    center_x = int(x + w * 0.5)
    center_y = int(y + h * 0.5)

    FOV = np.zeros((IMAGE_SIZE, IMAGE_SIZE, 3)).astype(np.uint8)
    cv2.circle(FOV, (int(IMAGE_SIZE / 2), int(IMAGE_SIZE / 2)), int(FOV_DIAMETER/2), (1, 1, 1), -1)

    img_arr = cv2.imread(img_path)
    img_expand = cv2.copyMakeBorder(img_arr, BORDER, BORDER, BORDER, BORDER, cv2.BORDER_CONSTANT, value=0)
    img_cropped = img_expand[center_y + BORDER - int(IMAGE_SIZE / 2):center_y + BORDER + int(IMAGE_SIZE / 2),
                  center_x + BORDER - int(IMAGE_SIZE / 2):center_x + BORDER + int(IMAGE_SIZE / 2), :]
    img_cropped = img_cropped * FOV
    # return img_cropped, FOV[:, :, 1], diameter
    return img_cropped, FOV[:, :, 1], diameter