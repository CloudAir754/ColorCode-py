import cv2
import numpy as np

def detect_green_diagonals(self):
    """
    从已处理的图像中提取对角线上的绿色色块，并可选地在原始图像上显示结果
    """
    # 初始化存储数组
    self.green_diagonals = []
    
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
            self.green_diagonals.append(( x , y , w , h ))
            
    
    # 按照x+y的值从小到大排序
    self.green_diagonals.sort(key=lambda point: point[0] + point[1])

    step_dia = 0

    # 在图像上绘制标记
    for (x, y, w, h) in self.green_diagonals:
        step_dia+=1
        
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
