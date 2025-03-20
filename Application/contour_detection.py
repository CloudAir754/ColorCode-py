import cv2
import numpy as np

def detect_contours(self):
    """这个函数主要作用是剔除无用边缘"""
    contours, _ = cv2.findContours(self.closed_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    img_size_x = self.target_size_x
    img_size_y = self.target_size_y  # 分别获取x和y方向的目标尺寸
    min_dim_x = img_size_x // self.min_screen_coef
    min_dim_y = img_size_y // self.min_screen_coef
    max_dim_x = img_size_x // self.max_screen_coef
    max_dim_y = img_size_y // self.max_screen_coef

    valid_contours = []  # 可用边缘
    quadrilaterals = []  # 内接四边形

    for cnt in contours:# 筛选每个轮廓
        if len(cnt) < 5:  # 至少需要5个点才能构成有效轮廓
            continue
        area = cv2.contourArea(cnt)
        if area < self.min_contour_area: # 最小轮廓面积限制
            continue

        try:
            rect = cv2.minAreaRect(cnt)
            # minAreaRect 计算的是外接矩形（旋转）
            box = cv2.boxPoints(rect)
            # 旋转矩形的四个点坐标
            box = np.intp(box)

            # 分别检查x和y方向是否超出图像边界
            if (box[:, 0] < 0).any() or (box[:, 0] >= img_size_x).any() or \
            (box[:, 1] < 0).any() or (box[:, 1] >= img_size_y).any():
                continue

            # 计算旋转矩形的宽度和高度
            width = max(np.linalg.norm(box[0] - box[1]), np.linalg.norm(box[2] - box[3]))
            height = max(np.linalg.norm(box[1] - box[2]), np.linalg.norm(box[3] - box[0]))

            # 判断宽度和高度是否在允许的范围内
            if not (min_dim_x <= width <= max_dim_x and min_dim_y <= height <= max_dim_y):
                continue
            

        except:
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
        

    self.contours = valid_contours
    self.quadrilaterals = self.sort_quad(quadrilaterals)  # 调用类的方法

    # 断点
    # cv2.waitKey()
    return

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
    """对 9 个四边形进行排序，从左到右：按照123第一排；456第二排；789第三排"""
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