import cv2
import numpy as np
from PIL import Image


def sum_3x3_center_padded(arr, x, y):
    padded_arr = np.zeros((arr.shape[0]+2, arr.shape[1]+2), dtype=arr.dtype)
    padded_arr[1:-1, 1:-1] = arr
    new_x, new_y = x + 1, y + 1
    region_sum = padded_arr[new_y - 1:new_y + 2, new_x - 1:new_x + 2].sum()
    return region_sum


def sum_3x3_kernel_padded(arr, x, y, kernel):
    padded_arr = np.zeros((arr.shape[0]+2, arr.shape[1]+2), dtype=arr.dtype)
    padded_arr[1:-1, 1:-1] = arr
    new_x, new_y = x + 1, y + 1
    region_sum = np.sum(padded_arr[new_y - 1:new_y + 2, new_x - 1:new_x + 2] * kernel)
    return region_sum


def is_cross_point_four(x,y,arr):
    kernel1 = np.array([[1, 0, 1], [0, 1, 0], [0, 1, 0]])
    kernel2 = np.array([[0, 1, 0], [1, 1, 1], [0, 0, 0]])
    kernel3 = np.array([[0, 1, 0], [0, 1, 1], [1, 0, 0]])
    kernel4 = np.array([[1, 0, 1], [0, 1, 0], [0, 0, 1]])
    kernels = [kernel1, kernel2, kernel3, kernel4]
    for k in kernels:
        k1 = k
        k2 = cv2.rotate(k1, cv2.ROTATE_90_CLOCKWISE)
        k3 = cv2.rotate(k2, cv2.ROTATE_90_CLOCKWISE)
        k4 = cv2.rotate(k3, cv2.ROTATE_90_CLOCKWISE)
        ks = [k1, k2, k3, k4]
        for kernel in ks:
            sum_value = sum_3x3_kernel_padded(arr, x, y, kernel)
            if sum_value == 4:
                return True
    return False


def is_cross_point_five(x,y,arr):
    kernel1 = np.array([[0, 1, 0], [1, 1, 1], [0, 0,1]])
    kernel2 = np.array([[0, 1, 0], [1, 1, 1], [1, 0, 0]])
    kernel3 = np.array([[1, 0, 1], [1, 1, 0], [0, 0, 1]])
    kernel4 = np.array([[1, 0, 1], [0, 1, 0], [0, 1, 1]])
    kernel5 = np.array([[1, 0, 1], [0, 1, 1], [0, 1, 0]])
    kernel6 = np.array([[1, 0, 1], [1, 1, 0], [0, 1, 0]])
    kernels = [kernel1, kernel2, kernel3, kernel4, kernel5, kernel6]
    for k in kernels:
        k1 = k
        k2 = cv2.rotate(k1, cv2.ROTATE_90_CLOCKWISE)
        k3 = cv2.rotate(k2, cv2.ROTATE_90_CLOCKWISE)
        k4 = cv2.rotate(k3, cv2.ROTATE_90_CLOCKWISE)
        ks = [k1, k2, k3, k4]
        for kernel in ks:
            sum_value = sum_3x3_kernel_padded(arr, x, y, kernel)
            if sum_value == 5:
                return True
    return False


def is_cross_point_six(x,y,arr):
    kernel1 = np.array([[1, 0, 1], [1, 1, 1], [0, 1, 0]])
    kernel2 = np.array([[0, 1, 0], [1, 1, 1], [1, 0, 1]])
    kernel3 = np.array([[1, 1, 0], [0, 1, 1], [1, 0, 1]])
    kernel4 = np.array([[1, 0, 1], [0, 1, 1], [0, 1, 1]])
    kernels = [kernel1, kernel2, kernel3, kernel4]
    for k in kernels:
        k1 = k
        k2 = cv2.rotate(k1, cv2.ROTATE_90_CLOCKWISE)
        k3 = cv2.rotate(k2, cv2.ROTATE_90_CLOCKWISE)
        k4 = cv2.rotate(k3, cv2.ROTATE_90_CLOCKWISE)
        ks = [k1, k2, k3, k4]
        for kernel in ks:
            sum_value = sum_3x3_kernel_padded(arr, x, y, kernel)
            if sum_value == 6:
                return True
    return False


def bifurcation_point(th_img, L):
    thh = th_img / 255
    thh = thh.astype(np.uint8)
    thh_copy = thh.copy()
    line_points = np.nonzero(thh)
    cross_points = []
    for n in range(len(line_points[0])):
        x = int(line_points[1][n])
        y = int(line_points[0][n])
        sum_value = sum_3x3_center_padded(thh, x, y)
        if sum_value <= 3:
            continue
        elif sum_value == 5:
            result = is_cross_point_five(x, y, thh)
            if result:
                cv2.circle(thh_copy, (x, y), 1, 0, -1)
                cross_points.append((x, y))
        elif sum_value >= 7:
            continue
        elif sum_value == 4:
            result = is_cross_point_four(x, y, thh)
            if result:
                cv2.circle(thh_copy, (x, y), 1, 0, -1)
                cross_points.append((x, y))
        elif sum_value == 6:
            result = is_cross_point_six(x, y, thh)
            if result:
                cv2.circle(thh_copy, (x, y), 1, 0, -1)
                cross_points.append((x, y))
        else:
            continue
    cross_points = list(set(cross_points))
    contours, hierarchy = cv2.findContours(thh_copy, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    for c in contours:
        if (c.size <= L):
            cv2.drawContours(thh_copy, [c], 0, 0, -1)
    contours, heirarchy = cv2.findContours(thh_copy, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    return contours, cross_points


def show_image(p):
    cv2.namedWindow('image', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('image', 1200, 1200)
    cv2.moveWindow('image', 40, 0)
    while True:
        cv2.imshow("image", p)
        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            break
    cv2.destroyAllWindows()


def get_skeleton_endpoints(skeleton):
    points = np.argwhere(skeleton == 255)
    points = points.astype(np.uint16)
    neighbors = [(-1, -1), (-1, 0), (-1, 1),
                 (0, -1), (0, 1),
                 (1, -1), (1, 0), (1, 1)]
    endpoints = []
    for r, c in points:
        count = 0
        for dr, dc in neighbors:
            if skeleton[r + dr, c + dc] == 255:
                count += 1
        if count == 1:
            endpoints.append((int(c), int(r)))
        elif count == 2:
            skeleton_copy = (skeleton.copy() / 255).astype(np.uint8)
            kernel1 = np.array([[1, 1, 0], [0, 0, 0], [0, 0, 0]])
            kernel2 = np.array([[0, 1, 1], [0, 0, 0], [0, 0, 0]])
            kernels = [kernel1, kernel2]
            for k in kernels:
                k1 = k
                k2 = cv2.rotate(k1, cv2.ROTATE_90_CLOCKWISE)
                k3 = cv2.rotate(k2, cv2.ROTATE_90_CLOCKWISE)
                k4 = cv2.rotate(k3, cv2.ROTATE_90_CLOCKWISE)
                ks = [k1, k2, k3, k4]
                for kernel in ks:
                    sum_value = sum_3x3_kernel_padded(skeleton_copy, c, r, kernel)
                    if sum_value == 2:
                        endpoints.append((int(c), int(r)))
    return endpoints


def calculate_square_distance(point1, point2):
    x1, y1 = point1
    x2, y2 = point2
    square_distance = (x2 - x1)**2 + (y2 - y1)**2
    return square_distance


def compare_points(reference_point, point1, point2):
    distance1 = calculate_square_distance(reference_point, point1)
    distance2 = calculate_square_distance(reference_point, point2)
    if distance1 < distance2:
        return [point1, point2]
    else:
        return [point2, point1]


def find_root(lines, ref_point):
    d = float('inf')
    l = None
    for line in lines:
        point1 = line["endpoint"][0]
        point2 = line["endpoint"][1]
        d1 = calculate_square_distance(ref_point, point1)
        d2 = calculate_square_distance(ref_point, point2)
        temp_d = min(d1, d2)
        if temp_d < d:
            d = temp_d
            l = line
    root_point = compare_points(ref_point, l["endpoint"][0], l["endpoint"][1])
    root_point.append(l["cnt"])
    root_point.append(len(l["cnt"]))
    return root_point


def find_other_element(two_element_list, known_element):
    if known_element not in two_element_list:
        raise ValueError("The known element is not in the list")
    # Since the list has only two elements, return the one that is not the known element
    return two_element_list[1] if two_element_list[0] == known_element else two_element_list[0]


def gen_tree(root, segments, leaf_points):
    tree = [root]
    endpoints = [root[1]]
    segments.remove([segment for segment in segments if root[0] in segment["endpoint"]][0])
    while endpoints:
        # show_image(thinned_copy)
        startpoint = endpoints.pop(0)
        segment_copy = segments.copy()
        for segment in segment_copy:
            if startpoint in segment["endpoint"]:
                endpoint = find_other_element(segment["endpoint"], startpoint)
                if endpoint not in leaf_points:
                    endpoints.append(endpoint)
                segments = [seg for seg in segments if seg["endpoint"] != segment["endpoint"]]
                tree.append((startpoint, endpoint, segment["cnt"], len(segment["cnt"])))
    return tree


class TreeNode:
    def __init__(self, coordinate):
        self.coordinate = coordinate
        self.children = []


def build_tree_from_data(data):
    nodes = {}
    for parent, child, cnt, distance in data:
        if parent not in nodes:
            nodes[parent] = TreeNode(parent)
        if child not in nodes:
            nodes[child] = TreeNode(child)
    for parent, child, cnt, distance in data:
        nodes[parent].children.append((nodes[child], cnt, distance))
    root_coordinate = data[0][0]
    return nodes[root_coordinate]


def print_tree(node, level=0):
    if node is not None:
        node_type = "Root" if level == 0 else "Leaf" if not node.children else "Intermediate"
        print(' ' * level * 2 + f"{node.coordinate} ({node_type})")
        for child, cnt, distance in node.children:
            print(' ' * (level * 2 + 2) + f'--{distance}--> {child.coordinate} (cnt={cnt})')
            print_tree(child, level + 1)


def find_longest_path(node):
    if not node.children:
        return (0, [node.coordinate])
    max_length = 0
    max_path = []
    for child, cnt, distance in node.children:
        child_length, child_path = find_longest_path(child)
        total_length = distance + child_length
        if total_length > max_length:
            max_length = total_length
            max_path = [node.coordinate] + child_path
    return (max_length, max_path)


def draw_path_on_image(path, data, image_size=1024):
    img = np.zeros((image_size, image_size), dtype=np.uint8)
    contour_dict = {}
    for parent, child, cnt, distance in data:
        contour_dict[(parent, child)] = cnt
    for i in range(len(path) - 1):
        start = path[i]
        end = path[i + 1]
        key = (start, end)
        if key in contour_dict:
            contour_points = contour_dict[key]
            cv2.drawContours(img, [contour_points], -1, 255, thickness=1)  # 255 是白色，1 是线宽
    kernel = np.ones((3, 3), np.uint8)
    img_dilate = cv2.dilate(img, kernel, iterations=5)
    # img_thinned = cv2.ximgproc.thinning(img_dilate)
    assert len(cv2.findContours(img_dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]) == 1, 'draw_path_on_image'
    return img_dilate


def gen_root(skeletons, endpoints, branchpoints, disk_point):
    cnt_list = []
    for i, cnt in enumerate(skeletons):
        d = {"cnt": None, "endpoint": []}
        for point in branchpoints + endpoints:
            distance = abs(cv2.pointPolygonTest(cnt, point, True))
            # print(distance)
            if distance < 2.5:
                d["cnt"] = cnt
                d["endpoint"].append(point)
        assert len(d["endpoint"]) == 2
        cnt_list.append(d)
    root_node = find_root(cnt_list, disk_point)
    tree_data = gen_tree(root_node, cnt_list, endpoints)
    root = build_tree_from_data(tree_data)
    return root, tree_data


