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
            
        contour = cv2.convexHull(cnt)  # 通过凸包矫正点序(轮廓点顺序错误​)
        area = cv2.contourArea(contour) # 计算面积

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

            
            width, height = rect[1]  # 直接使用 minAreaRect 的 width 和 height

            # 如果需要确保 width >= height
            width, height = sorted(rect[1], reverse=True)

            if not (min_dim_x <= width <= max_dim_x and min_dim_y <= height <= max_dim_y):
                continue

        except:
            continue
        
        valid_contours.append(cnt)
        quadrilaterals.append(box)


    print("有效内接四边形个数：")
    print(len(quadrilaterals))

    self.contours = valid_contours    
    self.quadrilaterals = quadrilaterals
    self.contours_ordered = self.sort_quad(quadrilaterals)  # 调用排序内部点


    if len(quadrilaterals) != 9:
        print("有效内接四边形不为9！！请检查超参数配置或检查图片")
        cv2.waitKey()     

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

    return sorted_quads