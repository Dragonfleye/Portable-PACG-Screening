import cv2
import json
import numpy as np
from PIL import Image
from scipy import optimize
from sympy import *
import os
import glob
import csv
from itertools import chain
import copy
import math
from tool import *


class Fit2Cur(object):
    def f_2(self, x, A, B, C):
        return A * x * x + B * x + C
    def fit_cur(self, x, y):
        coef = optimize.curve_fit(self.f_2, x, y)[0]
        return coef


class Fit1Cur(object):
    def f_1(self, x, A, B):
        return A * x + B
    def fit_cur(self, x, y):
        coef = optimize.curve_fit(self.f_1, x, y)[0]
        return coef


def fit_curve(points, start, end):
    assert end - start <= len(points), print("拟合曲线时，所选点长度超过原始数组长度")
    x = points[:, 0][start:end]
    y = points[:, 1][start:end]
    f2 = Fit2Cur()
    A, B, C = f2.fit_cur(x, y)
    return (A, B, C)


def fit_line(points, start, end):
    assert end - start <= len(points), print("拟合直线线时，所选点长度超过原始数组长度")
    x = points[:, 0][start:end]
    y = points[:, 1][start:end]
    f1 = Fit1Cur()
    A, B = f1.fit_cur(x, y)
    return (A, B)


def show_image(p):
    cv2.namedWindow('image', cv2.WINDOW_NORMAL)
    #cv2.moveWindow('image', 40, 0)
    while True:
        # ESC按下退出
        cv2.imshow("image", p)
        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            break
    cv2.destroyAllWindows()


def gen_three_band(img, oritation):
    def remove_small_patch(img, num_contours, fuse=False):
        contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # 按面积排序轮廓
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        # 选择面积最大的 num_contours 个轮廓
        largest_contours = contours[:num_contours]
        # 创建一个全零的图像
        largest_contours_image = np.zeros_like(img)
        # 绘制面积最大的轮廓
        cv2.drawContours(largest_contours_image, largest_contours, -1, 255, thickness=cv2.FILLED)
        if fuse:
            kernel = np.ones((5, 5), np.uint8)
            d_im = cv2.dilate(largest_contours_image, kernel, iterations=10)
            e_im = cv2.erode(d_im, kernel,iterations=10)
            e_contours, _ = cv2.findContours(e_im, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if len(e_contours) > 1:
                return (), None
            else:
                return np.squeeze(e_contours), e_im
        if num_contours == 1:
            return np.squeeze(largest_contours), largest_contours_image
        else:
            return _, largest_contours_image

    def clip(img, padding):
        contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        roi_x, roi_y, roi_w, roi_h = cv2.boundingRect(contours[0])
        roi = img[roi_y - padding:roi_y + roi_h + padding, roi_x - padding:roi_x + roi_w + padding]
        return roi, (roi_x - padding, roi_y - padding, roi_w + 2 * padding, roi_h + 2 * padding)

    def split_left_right(img, oritation):
        contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) == 1:
            return None, None
        centroids = []
        for contour in contours:
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                centroids.append((cX, cY))
            else:
                centroids.append((0, 0))
        if oritation == "H":
            sorted_contours = [contour for _, contour in sorted(zip(centroids, contours), key=lambda x: x[0][0])]
            left_contour_image = np.zeros_like(img)
            right_contour_image = np.zeros_like(img)
            cv2.drawContours(left_contour_image, [sorted_contours[0]], -1, 255, thickness=cv2.FILLED)
            cv2.drawContours(right_contour_image, [sorted_contours[1]], -1, 255, thickness=cv2.FILLED)
            return left_contour_image, right_contour_image
        else:
            sorted_contours = [contour for _, contour in sorted(zip(centroids, contours), key=lambda x: x[0][1])]
            left_contour_image = np.zeros_like(img)
            right_contour_image = np.zeros_like(img)
            cv2.drawContours(left_contour_image, [sorted_contours[0]], -1, 255, thickness=cv2.FILLED)
            cv2.drawContours(right_contour_image, [sorted_contours[1]], -1, 255, thickness=cv2.FILLED)
            return left_contour_image, right_contour_image
    up_band = np.zeros_like(img, dtype=np.uint8)
    up_band[img == 1] = 1
    ref_points, up_band_new = remove_small_patch(up_band, 1)
    length = max(cv2.minAreaRect(ref_points[:, np.newaxis, :])[1])
    if length < 800 and len(cv2.findContours(up_band, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]) >= 2:
        ref_points, up_band_new = remove_small_patch(up_band, 2, fuse=True)
        if len(ref_points) == 0:
            return (), ((), (), ())
    up_band_box, up_band_roi = clip(up_band_new, 10)
    down_band = np.zeros_like(img, dtype=np.uint8)
    down_band[img == 2] = 1
    _, down_band = remove_small_patch(down_band, 2)
    left_down_band, right_down_band = split_left_right(down_band, oritation)
    if left_down_band is None:
        return (), ((), (), ())
    left_down_band_box, left_down_band_roi = clip(left_down_band, 10)
    right_down_band_box, right_down_band_roi = clip(right_down_band, 10)
    return ref_points, ((up_band_box, up_band_roi), (left_down_band_box, left_down_band_roi), (right_down_band_box, right_down_band_roi))


def img_turn(img, ref_point, img_rot1, img_rot2, img_rot3):
    cnt = ref_point[:, np.newaxis, :].astype(np.int64)
    rows, cols = img_rot1.shape
    rect = cv2.minAreaRect(cnt)
    w = rect[1][0]
    h = rect[1][1]
    if w < h:
        rotation_angle = rect[-1] - 90
    else:
        rotation_angle = rect[-1]
    M = cv2.getRotationMatrix2D(rect[0], rotation_angle, 1)
    img_color = cv2.warpAffine(img, M, (cols, rows))
    res1 = cv2.warpAffine(img_rot1, M, (cols, rows))
    res2 = cv2.warpAffine(img_rot2, M, (cols, rows))
    res3 = cv2.warpAffine(img_rot3, M, (cols, rows))
    return img_color, res1, res2, res3


def gen_centerline(img):
    thinned = cv2.ximgproc.thinning(img)
    return thinned


def gen_centerline_origin(cl, size, roi_cor):
    C = np.zeros(size, np.uint8)
    C = Image.fromarray(C, mode="L")
    cl = Image.fromarray(cl, mode="L")
    C.paste(cl, box=(roi_cor[0]+1024, roi_cor[1]+1024))
    cl_new = np.array(C)
    return cl_new


def gen_coordinate(img):
    points = cv2.findNonZero(img)
    points = np.squeeze(points)
    points = points[np.argsort(points[:, 0]), :]
    coordinate = []
    index = 0
    while index < len(points) - 1:
        temp = []
        x = points[index][0]
        next_x = points[index + 1][0]
        if x != next_x:
            if len(temp) == 0:
                coordinate.append(points[index])
            else:
                temp.append(points[index])
                coordinate.append(np.mean(np.array(temp), axis=0, dtype=np.int32))
        else:
            temp.append(points[index])
        index += 1
    coordinate = np.array(coordinate)
    return coordinate


def extend_acre_fit(coor, **kwargs):
    for k,v in kwargs.items():
        if k == "L":
            if len(v) == 3:
                A, B, C = v
                start_x, start_y= coor[0]
                array_x = np.arange(start_x-250, start_x)
                array_y = np.array([round(A * i**2 + B * i + C) for i in array_x])
                points = np.stack([array_x, array_y],  axis=1)
                coor = np.concatenate([points, coor], axis=0)
            else:
                A, B = v
                start_x, start_y = coor[0]
                array_x = np.arange(start_x - 250, start_x)
                array_y = np.array([round(A * i  + B) for i in array_x])
                points = np.stack([array_x, array_y], axis=1)
                coor = np.concatenate([points, coor], axis=0)
        if k == "R":
            if len(v) == 3:
                A, B, C = v
                start_x, start_y = coor[-1]
                array_x = np.arange(start_x+1, start_x+250)
                array_y = np.array([round(A * i**2 + B * i + C) for i in array_x])
                points = np.stack([array_x, array_y], axis=1)
                coor = np.concatenate([coor, points], axis=0)
            else:
                A, B = v
                start_x, start_y = coor[-1]
                array_x = np.arange(start_x + 1, start_x + 250)
                array_y = np.array([round(A * i  + B ) for i in array_x])
                points = np.stack([array_x, array_y], axis=1)
                coor = np.concatenate([coor, points], axis=0)
    return coor


def extend_acre_dev(coor, lr=0, **kwargs):
    for k,v in kwargs.items():
        if k == "L":
            d = v
            start_x, start_y= coor[0]
            array_x = np.arange(start_x-350, start_x)
            array_y = np.full(array_x.shape, start_y)
            array_y = np.array([round(i-(d * (index* lr+1)) *(index+1)) for index, i in enumerate(array_y)])[::-1]
            points = np.stack([array_x, array_y],  axis=1)
            coor = np.concatenate([points, coor], axis=0)
        if k == "R":
            d = v
            start_x, start_y = coor[-1]
            array_x = np.arange(start_x+1, start_x+350)
            array_y = np.full(array_x.shape, start_y)
            array_y = np.array([round(i + (d * (index* lr+1)) * (index + 1)) for index, i in enumerate(array_y)])
            points = np.stack([array_x, array_y], axis=1)
            coor = np.concatenate([coor, points], axis=0)
    return coor


def cal_mean_deriv(array, start, end):
    if 1.5 * (end-start) > len(array):
        return 0
    x = [i[0] for i in array]
    y = [i[1] for i in array]
    diff_x = []
    for i, j in zip(x[0::], x[1::]):
        diff_x.append(j - i)
    diff_y = []
    for i, j in zip(y[0::], y[1::]):
        diff_y.append(j - i)
    slopes = []
    for i in range(len(diff_y)):
        slopes.append(diff_y[i] / diff_x[i])
    deriv = []
    for i, j in zip(slopes[0::], slopes[1::]):
        deriv.append((0.5 * (i + j)))
    deriv.insert(0, slopes[0])
    deriv.append(slopes[-1])
    return np.mean(np.array(deriv)[start:end])


def conect_acre(point1, point2):
    if np.min(point1[:,0]) < np.min(point2[:,0]):
        l_point = point1
        r_point = point2
    else:
        l_point = point2
        r_point = point1
    start_point = l_point[-1]
    end_point = r_point[0]
    point_x = np.arange(start_point[0]+1, end_point[0])
    if len(point_x) == 0:
        return 0, "虹膜的两条光带重叠"
    d = (end_point[1] - start_point[1]) / len(point_x)
    point_y = np.full(point_x.shape, start_point[1])
    point_y = np.array([round(y+d*(index+1)) for index, y in enumerate(point_y)])
    points = np.stack([point_x, point_y],  axis=1)
    return np.concatenate([l_point,  points, r_point], axis=0)


def is_continuous(points):
    d = [int(points[i] - points[i-1]) for i in range(1, len(points))]
    res = np.where(np.array(d) == 1, 0, 1)
    if np.sum(res) == 0:
        return True
    else:
        return False


def detect_cross_point(pic):
    left_cross_point = 0
    right_cross_point = 0
    for i in range(int(pic.shape[1]*0.5), 0, -1):
        line = pic[:, i]
        if np.sum(line) == 0:
            continue
        practical_points = np.where(line ==255)[0]
        res = is_continuous(practical_points)
        if res:
            left_cross_point = i
            break
    for i in range(int(pic.shape[1]*0.5), pic.shape[1]):
        line = pic[:, i]
        if np.sum(line) == 0:
            continue
        practical_points = np.where(line ==255)[0]
        res = is_continuous(practical_points)
        if res:
            right_cross_point = i
            break
    return left_cross_point, right_cross_point


def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx


def compute_horizontal_diameter(points):
    diameter_list = []
    left_points_list = []
    right_points_list = []
    x = points[:, 0]
    y = points[:, 1]
    up_y = np.max(y)
    down_y = np.min(y)
    mid_x = np.mean(x)
    points = np.array(list(zip(x, y)))
    left_points = points[points[:, 0] <= mid_x]
    right_points = points[points[:, 0] > mid_x]
    for i in range(down_y, up_y):
        left_p = left_points[find_nearest(left_points[:,1], i)]
        right_p = right_points[find_nearest(right_points[:,1], i)]
        diameter = right_p[0] - left_p[0]
        diameter_list.append(diameter)
        left_points_list.append(left_p)
        right_points_list.append(right_p)
    max_diameter = np.max(np.array(diameter_list))
    max_diameter_index = np.argmax(np.array(diameter_list))
    left_point = left_points_list[max_diameter_index]
    right_point = right_points_list[max_diameter_index]
    return max_diameter, (left_point[0], round((left_point[1]+right_point[1])/2)), (right_point[0], round((left_point[1]+right_point[1])/2))


def compute_closed_area(size, cl_up, cl_left_down, cl_right_down):
    cnt = np.concatenate([cl_up,cl_right_down[::-1], cl_left_down[::-1]], axis=0)[:,np.newaxis,:]
    area = cv2.contourArea(cnt)
    return area


def abstract_info(pic, diameter_cs):
    area_cs = (math.pi * diameter_cs ** 2) / 4
    contours, _ = cv2.findContours(pic, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) > 1:
        return 0, "前房被分为多于1的区域："
    if len(contours) == 0:
        return 0, "未发现有效轮廓："
    contour_ac = contours[0]
    area_ac = cv2.contourArea(contour_ac)
    if area_ac < 50000:
        return 0, "前房面积过小："
    ac2cs = area_ac / area_cs

    leftmost = tuple(contour_ac[contour_ac[:, :, 0].argmin()][0])
    rightmost = tuple(contour_ac[contour_ac[:, :, 0].argmax()][0])
    left_break_point = int((rightmost[0] - leftmost[0]) / 5) + leftmost[0]
    right_break_point = rightmost[0] - int((rightmost[0] - leftmost[0]) / 5)

    left_ac_index = np.where(cv2.findNonZero(pic)[:, :, 0] == left_break_point)[0]
    left_ac_points = cv2.findNonZero(pic)[left_ac_index]
    left_ac_up_point = left_ac_points[left_ac_points[:, :, 1].argmin()][0]
    left_ac_down_point = left_ac_points[left_ac_points[:, :, 1].argmax()][0]

    right_ac_index = np.where(cv2.findNonZero(pic)[:, :, 0] == right_break_point)[0]
    right_ac_points = cv2.findNonZero(pic)[right_ac_index]
    right_ac_up_point = right_ac_points[right_ac_points[:, :, 1].argmin()][0]
    right_ac_down_point = right_ac_points[right_ac_points[:, :, 1].argmax()][0]

    left_ac = copy.deepcopy(pic)
    right_ac = copy.deepcopy(pic)

    left_ac[:, left_break_point + 1:] = 0
    right_ac[:, :right_break_point] = 0

    cv2.line(left_ac, left_ac_up_point, left_ac_down_point, [255, 255, 255])
    cv2.line(right_ac, right_ac_up_point, right_ac_down_point, [255, 255, 255])
    left_ac_contour = \
    cv2.findContours(cv2.threshold(left_ac, 30, 255, 0)[-1], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
    left_ac_area = cv2.contourArea(left_ac_contour[0])
    right_ac_contour = \
    cv2.findContours(cv2.threshold(right_ac, 30, 255, 0)[-1], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
    right_ac_area = cv2.contourArea(right_ac_contour[0])

    left_ac[:, :leftmost[0]+10] = 0
    left_ac[:, left_break_point -10:] = 0

    right_ac[:, rightmost[0] - 10:] = 0
    right_ac[:, :right_break_point + 10] = 0

    left_ac_contours, _ = cv2.findContours(left_ac, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    left_ac_coses = [cv2.fitLine(cnt, cv2.DIST_L2, 0, 0.01, 0.01)[1][0] for cnt in left_ac_contours]
    left_ac_angles = [math.acos(cos)*180/math.pi for cos in left_ac_coses]
    left_ac_angle = abs(left_ac_angles[0] - left_ac_angles[1])

    right_ac_contours, _ = cv2.findContours(right_ac, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    right_ac_coses = [cv2.fitLine(cnt, cv2.DIST_L2, 0, 0.01, 0.01)[1][0] for cnt in right_ac_contours]

    right_ac_angles = [math.acos(cos) * 180 / math.pi for cos in right_ac_coses]
    right_ac_angle = abs(right_ac_angles[0] - right_ac_angles[1])

    pic_copy = copy.deepcopy(pic)
    pic_copy[:, :leftmost[0] + 10] = 0
    pic_copy[:, rightmost[0] - 10:] = 0
    contours_up_down, _ = cv2.findContours(pic_copy, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    if len(contours_up_down) != 2:
        return 0, "角膜和虹膜的光带不为2"

    lengths = [cv2.arcLength(cnt, True) / 2 for cnt in contours_up_down]
    chord = min(lengths)
    arc = max(lengths)
    a2c = arc / chord
    adc = arc - chord
    center_area = area_ac - left_ac_area - right_ac_area
    return [area_ac, ac2cs, int((left_ac_area+right_ac_area)/2), center_area, int((left_ac_angle+right_ac_angle)/2), a2c, adc]


def log_info(file, head, info):
    if not os.path.exists(file):
        with open(file, mode="w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(head)
            writer.writerow(info)
    else:
        with open(file, mode="a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(info)


def read_err(file):
    if not os.path.exists(file):
        return []
    else:
        with open(file, mode="r") as f:
            reader = csv.reader(f)
            head = next(reader)
            error_info = ["-".join(lin[0].split(os.sep)[5:]) for lin in reader]
        return error_info


def trans_cor(w, cor):
    x = cor[:, 1][:, np.newaxis]
    y = w - cor[:, 0][:, np.newaxis]
    points = np.concatenate((x, y), axis=1)
    return points


def abstract_label(csv_file, img):
    img_num = img[:6]
    eye = img[6]
    img_name = img_num + "-" + eye
    limb_diameter = []
    with open(csv_file, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        head = next(reader)
        for lin in reader:
            limb_diameter.append({"img_name": lin[0].split(".")[0], "label": int(lin[-1])})
    label = [d["label"] for d in limb_diameter if img_name == d["img_name"]][0]
    return label


def generate_skeleton_line(skeleton_image, points):
    def remove_small_patch(img, num_contours):
        contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        largest_contours = contours[:num_contours]
        largest_contours_image = np.zeros_like(img)
        cv2.drawContours(largest_contours_image, largest_contours, -1, 255, thickness=20)
        kernel = np.ones((3, 3), np.uint8)
        largest_contours_image = cv2.erode(largest_contours_image, kernel, iterations=5)
        return np.squeeze(largest_contours), largest_contours_image

    for i in range(len(points)):
        pt = points[i]
        cv2.circle(skeleton_image, pt, 10, 255, -1)
    contours, _ = cv2.findContours(skeleton_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) != 1:
        _, skeleton_image = remove_small_patch(skeleton_image, 1)
        return skeleton_image
    return skeleton_image


def find_end(points, ref_point):
    def calculate_square_distance(point1, point2):
        x1, y1 = point1
        x2, y2 = point2
        square_distance = (x2 - x1) ** 2 + (y2 - y1) ** 2
        return square_distance
    point = None
    d = float('inf')
    for p in points:
        temp_d = calculate_square_distance(ref_point, p)
        if temp_d < d:
            d = temp_d
            point = p
    return point


def find_nearest_line(skeletons, point):
    l = None
    d = float('inf')
    for lin in skeletons:
        temp_d = abs(cv2.pointPolygonTest(lin, point, True))
        if temp_d < d:
            d = temp_d
            l = lin
    return l


def clip_branch(skeleton, oritation=False):
    end_points = get_skeleton_endpoints(skeleton)
    if len(end_points) == 2:
        return skeleton
    elif len(end_points) == 3:
        skeletons, branchpoints = bifurcation_point(skeleton, L=0)
        assert len(branchpoints) == 1
        skeletons = sorted(skeletons, key=len, reverse=True)
        largest_contours = skeletons[:2]
        largest_contours_image = np.zeros_like(skeleton)
        cv2.drawContours(largest_contours_image, largest_contours, -1, 255, thickness=cv2.FILLED)
        kernel = np.ones((3, 3), np.uint8)
        img_dilate = cv2.dilate(largest_contours_image, kernel, iterations=5)
        img_thinned = cv2.ximgproc.thinning(img_dilate)
        assert len(cv2.findContours(img_thinned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]) == 1
        return img_thinned
    elif len(end_points) > 3:
        h, w = skeleton.shape
        skeletons, branchpoints = bifurcation_point(skeleton, L=0)
        if len(branchpoints) == 0:  # 遇到了十字形的分叉，不做处理了
            return skeleton
        else:
            if oritation == "U" or oritation == "R":
                start_point = find_end(end_points, (0, h))
                end_point = find_end(end_points, (w,h))
            elif oritation == "D" or oritation == "L":
                start_point = find_end(end_points, (0,0))
                end_point = find_end(end_points, (w, 0))
            else:
                start_point = find_end(end_points, (0, int(h / 2)))
                end_point = find_end(end_points, (w, int(h / 2)))
            end_points.remove(start_point)
            end_points.remove(end_point)
            largest_contours_image = skeleton.copy()
            for ep in end_points:
                lin = find_nearest_line(skeletons, ep)
                cv2.drawContours(largest_contours_image, [lin], 0, 0, thickness=cv2.FILLED)
            kernel = np.ones((3, 3), np.uint8)
            img_dilate = cv2.dilate(largest_contours_image, kernel, iterations=5)
            img_thinned = cv2.ximgproc.thinning(img_dilate)
            assert len(cv2.findContours(img_thinned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]) == 1
            if len(get_skeleton_endpoints(skeleton)) > 2:
                clip_branch(img_thinned)
            return img_thinned


def compute(img_color, img_band, limbus_diameter, pic_info):
    case, eye, light, bias = pic_info
    h = img_band.shape[0]
    w = img_band.shape[1]

    ref_points, (img_up_info, img_left_down_info, img_right_down_info) = gen_three_band(img_band, light)
    if len(ref_points) == 0:
        return "角膜光带被分为两部分或虹膜光带只识别到一条"

    padded_image = np.zeros((4096, 4096, 3), dtype=np.uint8)
    padded_image[1024:3072, 1024:3072, :] = img_color
    img_color = padded_image

    if light == "V":
        img_up_info, img_left_down_info, img_right_down_info = \
            [(cv2.rotate(ii[0], cv2.ROTATE_90_COUNTERCLOCKWISE),
               (ii[1][1], w-ii[1][2]-ii[1][0], ii[1][3], ii[1][2]))
               for ii in [img_up_info, img_left_down_info, img_right_down_info]]
        img_color = cv2.rotate(img_color, cv2.ROTATE_90_COUNTERCLOCKWISE)
        ref_points = trans_cor(w, ref_points)

    cl_up, cl_left_down, cl_right_down =\
        [gen_centerline(i) for i in [img_up_info[0], img_left_down_info[0], img_right_down_info[0]]]
    cl_up = clip_branch(cl_up, oritation=bias)
    cl_left_down = clip_branch(cl_left_down)
    cl_right_down = clip_branch(cl_right_down)

    cl_up, cl_left_down, cl_right_down = \
        [gen_centerline_origin(c, (2 * h, 2 * w), (i[0], i[1])) for c, i in [(cl_up, img_up_info[1]),
                                            (cl_left_down, img_left_down_info[1]),
                                           (cl_right_down, img_right_down_info[1])]]
    img_color, cl_up, cl_left_down, cl_right_down = \
        img_turn(img_color, ref_points + 1024, cl_up, cl_left_down, cl_right_down)


    coor_up = gen_coordinate(cl_up)
    coor_left_down = gen_coordinate(cl_left_down)
    coor_right_down = gen_coordinate(cl_right_down)

    coor_up_clipped = coor_up[50:-50]
    coor_left_down_clipped = coor_left_down[50:-30]
    coor_right_down_clipped = coor_right_down[30:-50]

    deriv_up_left = cal_mean_deriv(coor_up_clipped, 0, 50)
    if deriv_up_left == 0:
        return "计算梯度时，所选点长度超过原始数组长度"

    deriv_up_right = cal_mean_deriv(coor_up_clipped, len(coor_up_clipped)-50, len(coor_up_clipped))
    if deriv_up_right == 0:
        return "计算梯度时，所选点长度超过原始数组长度"

    if deriv_up_right * deriv_up_left >= 0:
        coor_up_clipped = coor_up_clipped[50:-50]
        deriv_up_left = cal_mean_deriv(coor_up_clipped, 0, 50)
        if deriv_up_left == 0:
            return "计算梯度时，所选点长度超过原始数组长度"
        deriv_up_right = cal_mean_deriv(coor_up_clipped, len(coor_up_clipped) - 50, len(coor_up_clipped))
        if deriv_up_right == 0:
            return "计算梯度时，所选点长度超过原始数组长度"

    deriv_left_down = cal_mean_deriv(coor_left_down_clipped, 0, 50)
    if deriv_left_down == 0:
        return "计算梯度时，所选点长度超过原始数组长度"

    deriv_right_down = cal_mean_deriv(coor_right_down_clipped, len(coor_right_down_clipped) - 50, len(coor_right_down_clipped))
    if deriv_right_down == 0:
        return "计算梯度时，所选点长度超过原始数组长度"

    new_coor_up = extend_acre_dev(coor_up_clipped, lr=0.02, L=deriv_up_left, R=deriv_up_right)
    if np.min(new_coor_up[:, 1]) < 0:
        new_coor_up = np.delete(new_coor_up, np.where(new_coor_up[:, 1] < 0)[0], axis=0)
    if np.max(new_coor_up[:, 1]) > 2 * h:
        new_coor_up = np.delete(new_coor_up, np.where(new_coor_up[:, 1] > 2 * h)[0], axis=0)
    new_coor_left_down = extend_acre_dev(coor_left_down_clipped, lr=0.001, L=deriv_left_down)
    if np.min(new_coor_left_down[:, 1]) < 0:
        new_coor_left_down = np.delete(new_coor_left_down,
                                       np.where(new_coor_left_down[:, 1] < 0)[0], axis=0)
    if np.max(new_coor_left_down[:, 1]) > 2 * h:
        new_coor_left_down = np.delete(new_coor_left_down,
                                       np.where(new_coor_left_down[:, 1] > 2 * h)[0], axis=0)
    new_coor_right_down = extend_acre_dev(coor_right_down_clipped, lr=0.001, R=deriv_right_down)
    if np.min(new_coor_right_down[:, 1]) < 0:
        new_coor_right_down = np.delete(new_coor_right_down,
                                        np.where(new_coor_right_down[:, 1] < 0)[0], axis=0)
    if np.max(new_coor_left_down[:, 1]) > 2 * h:
        new_coor_right_down = np.delete(new_coor_right_down,
                                        np.where(new_coor_right_down[:, 1] > 2 * h)[0], axis=0)

    new_coor_down = conect_acre(new_coor_left_down, new_coor_right_down)
    if isinstance(new_coor_down, tuple):
        return new_coor_down[-1]
    if np.min(new_coor_down)< 0 or np.max(new_coor_down[:, 0]) >= 2 * w or np.max(new_coor_down[:, 1]) >= 2 * w:
        return "延长曲线超出图片范围"
    if np.min(new_coor_up)<0 or np.max(new_coor_up[:, 0]) >= 2 * w or np.max(new_coor_up[:, 1]) >= 2 * w:
        return "延长曲线超出图片范围 "

    background = np.zeros((2 * h, 2 * w)).astype(np.uint8)
    background = generate_skeleton_line(background, new_coor_down)

    background = generate_skeleton_line(background, new_coor_up)
    background = background.astype(np.uint8)

    left_cross_point, right_cross_point = detect_cross_point(background)
    background[:, 0:left_cross_point-4] = 0
    background[:, right_cross_point+5:] = 0

    closed_area = compute_closed_area((h, w), coor_up_clipped, coor_left_down_clipped, coor_right_down_clipped)
    info = abstract_info(background, limbus_diameter)
    if info[0] == 0:
        return info[-1]
    else:
        return [closed_area] + info
