from model.limbus_model import limbus_infer
from model.cornea_iris_band_model import cornea_iris_band_infer
from compute_parameter import compute
import os


def abstract_pic_info(img):
    img_prefix = img.split(os.sep)[-1].split('.')[0]
    eye = img_prefix[-3]
    light = img_prefix[-2]
    bias = img_prefix[-1]
    case = img_prefix[:6]
    return case, eye, light, bias


def main(img_path):
    print(f"Starting to process the image {img_path.split(os.sep)[-1]}")
    img_info = abstract_pic_info(img_path)
    preprocessed_image, mask, limbus_diameter = limbus_infer(img_path)
    band_image = cornea_iris_band_infer(preprocessed_image, mask)
    parameters = compute(preprocessed_image, band_image, limbus_diameter, img_info)
    if isinstance(parameters, list):
        print("Parameter I(closed area):", parameters[0])
        print("Parameter II(central anterior chamber section area:", parameters[1])
        print("Parameter III(ratio of central anterior chamber section area to limbus area):", parameters[2])
        print("Parameter IV(anterior chamber angle area):", parameters[3])
        print("Parameter V(central area of the central anterior chamber section:", parameters[5])
        print("Parameter VI(anterior chamber angle):", parameters[6])
        print("Parameter VII(ratio of lengths of the corneal band to iris band):", parameters[6])
        print("Parameter VIII(difference in lengths between the corneal band and iris band):", parameters[7])
    else:
        print(parameters)


if __name__ == '__main__':
    img_path = "sample/000377LHU.jpg"
    main(img_path)