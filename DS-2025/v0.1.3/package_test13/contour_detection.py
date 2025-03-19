import cv2
import numpy as np

def detect_contours(self):
    """这个函数主要作用是剔除无用边缘"""
    contours, _ = cv2.findContours(self.closed_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    img_size = self.target_size
    min_dim = img_size // self.min_screen_coef
    max_dim = img_size // self.max_screen_coef

    valid_contours = []  # 可用边缘
    quadrilaterals = []  # 内接四边形

    for cnt in contours:
        if len(cnt) < 5:  # 至少需要5个点才能构成有效轮廓
            continue

        area = cv2.contourArea(cnt)
        if area < self.min_contour_area:
            continue

        try:
            rect = cv2.minAreaRect(cnt)
            box = cv2.boxPoints(rect)
            box = np.intp(box)

            if (box < 0).any() or (box >= img_size).any():
                continue
        except:
            continue

        x, y, w, h = cv2.boundingRect(cnt)
        if not (min_dim < w < max_dim and min_dim < h < max_dim):
            continue

        valid_contours.append(cnt)
        quadrilaterals.append(box)

    print("有效边缘个数：")
    print(len(valid_contours))
    print("有效内接四边形个数：")
    print(len(quadrilaterals))

    if len(quadrilaterals) != 9:
        print("有效内接四边形不为9！！请检查超参数配置或检查图片")
        cv2.waitKey()
        return []

    self.contours = valid_contours
    self.quadrilaterals = self.sort_quad(quadrilaterals)  # 调用类的方法

    return quadrilaterals

def sort_quad(self, quadrilaterals):
    """将每个四边形的顶点按照左上、右上、右下、左下的顺序排序。"""
    sorted_quads = []

    for quad in quadrilaterals:
        quad = np.array(quad)
        sorted_by_x = quad[np.argsort(quad[:, 0])]
        left_points = sorted_by_x[:2]
        right_points = sorted_by_x[2:]

        left_points = left_points[np.argsort(left_points[:, 1])]
        top_left, bottom_left = left_points[0], left_points[1]

        right_points = right_points[np.argsort(right_points[:, 1])]
        top_right, bottom_right = right_points[0], right_points[1]

        sorted_quad = np.array([top_left, top_right, bottom_right, bottom_left])
        sorted_quads.append(sorted_quad)

    sorted_quadrilaterals = self.sort_quadrilaterals(sorted_quads)
    return sorted_quadrilaterals

def sort_quadrilaterals(self, points_list):
    """对 9 个四边形进行排序，按照从左到右、从上到下的顺序排列。"""
    quad_centers = []
    for quad in points_list:
        cx = sum(p[0] for p in quad) / 4
        cy = sum(p[1] for p in quad) / 4
        quad_centers.append((cx, cy))

    a_index = min(range(9), key=lambda i: quad_centers[i][0] + quad_centers[i][1])
    A = points_list[a_index]

    top_avg = (A[0][1] + A[1][1]) / 2
    bottom_avg = (A[2][1] + A[3][1]) / 2

    same_row = []
    for i in range(9):
        if i == a_index:
            continue
        cx, cy = quad_centers[i]
        if top_avg <= cy <= bottom_avg:
            same_row.append((cx, i))

    same_row.sort(key=lambda x: x[0])
    B_index = same_row[0][1]
    C_index = same_row[1][1]

    remaining = set(range(9)) - {a_index, B_index, C_index}

    B = points_list[B_index]
    left_x = (B[0][0] + B[3][0]) / 2
    right_x = (B[1][0] + B[2][0]) / 2

    left_col = []
    mid_col = []
    right_col = []
    for i in remaining:
        cx, cy = quad_centers[i]
        if cx < left_x:
            left_col.append((cy, i))
        elif cx > right_x:
            right_col.append((cy, i))
        else:
            mid_col.append((cy, i))

    left_col.sort(key=lambda x: x[0])
    mid_col.sort(key=lambda x: x[0])
    right_col.sort(key=lambda x: x[0])

    if len(left_col) < 2 or len(mid_col) < 2 or len(right_col) < 2:
        print("未足额找到第二行和第三行的方块")
        print(left_col)
        print(mid_col)
        print(right_col)
        print("=======================================")

    indices = [a_index, B_index, C_index]
    indices.append(left_col[0][1])
    indices.append(mid_col[0][1])
    indices.append(right_col[0][1])
    indices.append(left_col[1][1])
    indices.append(mid_col[1][1])
    indices.append(right_col[1][1])

    ordered = [points_list[i] for i in indices if i is not None]
    self.contours_ordered = ordered
    return ordered