import cv2
import numpy as np

def detect_green_diagonals(self):
    """
    从已处理的图像中提取对角线上的绿色色块，并可选地在原始图像上显示结果
    """
    
    # 检查前置条件
    if not hasattr(self, 'Sized_img') or not self.contours_ordered:
        print("[WARN] 缺少预处理图像或四边形数据")
        return
    
    # 获取已调整尺寸的图像
    img_diag = self.Sized_img.copy() 
    
    for i in range(len(self.contours_ordered)):
        quad = self.contours_ordered[i]
                
        # 获取四边形外接矩形
        x, y, w, h = cv2.boundingRect(quad)
        
        
        center_x = int(x + w * (0.5 - self.center_factor / 2))
        center_y = int(y + h * (0.5 - self.center_factor / 2))
        center_w = int(w * self.center_factor)
        center_h = int(h * self.center_factor)
        
        # 提取ROI区域
        roi = img_diag[center_y:center_y + center_h, 
                 center_x:center_x + center_w]
        
        # 转换为HSV空间
        hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        
        # 计算平均HSV值
        h_mean = int(np.mean(hsv_roi[:, :, 0])) * 2  # 0-360
        s_mean = int(np.mean(hsv_roi[:, :, 1])) / 255  # 0-1
        v_mean = int(np.mean(hsv_roi[:, :, 2])) / 255  # 0-1
        
        # 颜色分类
        color = self.classify_color((h_mean, s_mean, v_mean))
        
        # 仅当确实是绿色时才存储
        if color == "Green":
            self.green_diagonals.append(quad)
            
    
    # 按照x+y的值从小到大排序
    # print(self.green_diagonals)
    self.green_diagonals.sort(key=lambda point: point[0][0] + point[0][1]) # 左上角坐标 x+y

    step_dia = 0

    # 在图像上绘制标记
    for quad_tmp in self.green_diagonals:
        step_dia+=1
        x, y, w, h = cv2.boundingRect(quad_tmp)
        
        center_x = int(x + w * (0.5 - self.center_factor / 2))
        center_y = int(y + h * (0.5 - self.center_factor / 2))
        center_w = int(w * self.center_factor)
        center_h = int(h * self.center_factor)


        cv2.circle(img_diag, (center_x + center_w//2, center_y + center_h//2), 
                    5, (0, 255, 0), -1)
        # 绘制取样区域矩形
        cv2.rectangle(img_diag, (center_x, center_y),
                        (center_x + center_w, center_y + center_h),(0, 255, 0), 2)
        # 添加文本标签
        cv2.putText(img_diag, f"Green {step_dia}",(center_x, center_y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)


    # 验证是否找到三个绿色块
    if len(self.green_diagonals) != 3:
        print(f"[WARN] 只找到{len(self.green_diagonals)}个绿色对角线色块")
    
    # 显示结果图像
    self.visualize_process("Green Diagonals Detection", img_diag)

    return


def locate_nine(self):
    """
    计算每行和每列的左右边缘和上下边缘
    """
    
    # 检查是否有绿色对角线
    if not hasattr(self, 'green_diagonals') or not self.green_diagonals:
        print("[WARN] 没有检测到绿色对角线")
        return

    # 初始化行列的边缘序列
    row_boundaries = [] # 行 （上边界，下边界）
    col_boundaries = [] # 列 （左边界，右边界）
    


    for quad in self.green_diagonals:
        # 获取四边形的边界框
        x, y, w, h = cv2.boundingRect(quad)
        
        # 获取当前四边形的左边界和右边界
        left_edge = x
        right_edge = x + w
        
        # 获取当前四边形的上边界和下边界
        top_edge = y
        bottom_edge = y + h
        
        row_boundaries.append((top_edge,bottom_edge))
        col_boundaries.append((left_edge,right_edge))
       
    # 存储所有中心坐标的列表
    center_points = []
    # 用来存储每个轮廓的行列编号
    contour_positions = []
    
    for quad in self.contours_ordered:
        # 获取四边形的边界框
        x, y, w, h = cv2.boundingRect(quad)
        
        # 计算中心点坐标
        center_x = int(x + w / 2)
        center_y = int(y + h / 2)
        
        # 存储中心坐标
        center_points.append((center_x, center_y))

        # 判断该中心点属于哪一行
        row = None
        for idx, (top_edge, bottom_edge) in enumerate(row_boundaries):
            if top_edge <= center_y <= bottom_edge:
                row = idx
                break
        
        # 判断该中心点属于哪一列
        col = None
        for idx, (left_edge, right_edge) in enumerate(col_boundaries):
            if left_edge <= center_x <= right_edge:
                col = idx
                break
        
        # 将行列编号存储在 contour_positions 中
        contour_positions.append((row, col))

     # 输出每个轮廓的行列编号
    print("每个轮廓的行列编号:")
    for idx, (row, col) in enumerate(contour_positions):
        print(f"轮廓 {idx + 1} 的行列编号: 行 {row}, 列 {col}")


