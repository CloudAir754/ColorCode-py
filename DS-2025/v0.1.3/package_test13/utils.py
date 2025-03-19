import cv2
import numpy as np

def detect_stretch_ratio(self):
    """检测拉伸比率，并检查波动是否过大。"""
    if not self.contours_ordered:
        raise ValueError("没有检测到有效的四边形，无法计算拉伸比率。")
    
    ratios = []
    
    # 计算每个四边形的长宽比
    for quad in self.contours_ordered:
        # 计算四边形的宽度和高度
        width = np.linalg.norm(quad[0] - quad[1])  # 左上到右上的距离
        height = np.linalg.norm(quad[1] - quad[2])  # 右上到右下的距离
        
        # 计算长宽比 x/y
        ratio = width / height
        ratios.append(ratio)
    
    # 计算平均长宽比
    mean_ratio = np.mean(ratios)
    
    # 检查每个比率与均值的差异
    for i, ratio in enumerate(ratios):
        deviation = abs(ratio - mean_ratio) / mean_ratio
        if deviation > self.HP_ts_radio:
            print((f"第 {i+1} 个四边形的拉伸比率 {ratio:.2f} 与均值 {mean_ratio:.2f} 的差异超过阈值 {self.HP_ts_radio*100:.0f}%。"))
            # raise ValueError(f"第 {i+1} 个四边形的拉伸比率 {ratio:.2f} 与均值 {mean_ratio:.2f} 的差异超过阈值 {self.HP_ts_radio*100:.0f}%。")
    
    self.radio_stretch = mean_ratio
    return mean_ratio

def visualization_detect_contours(self):
    """最终识别成果，可视化"""
    if not self.show_steps:
        # 不执行该函数
        return
    
    valid_contours = self.contours
    quadrilaterals = self.quadrilaterals

    # 改进的可视化
    contour_image = cv2.resize(self.image.copy(), (self.target_size, self.target_size))
    
    # 绘制原始轮廓
    cv2.drawContours(contour_image, valid_contours, -1, (0, 255, 0), 2)
    
    # 绘制四边形（带有效性检查）
    step_vi = 0
    for quad in quadrilaterals:
        step_vi += 1
        # 绘制四边形边界
        cv2.polylines(contour_image, [quad], True, (0, 0, 255), 2)  # 蓝色外接矩形
        
        # 标注坐标点（可选）
        for j, (x, y) in enumerate(quad):
            cv2.circle(contour_image, (x, y), 3, (255, 0, 0), -1)
            cv2.putText(contour_image, f"{j}", (x + 5, y + 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        # 绘制 step 信息
        # 计算四边形的中心点
        x, y, w, h = cv2.boundingRect(quad)
        center_x = x + w // 2
        center_y = y + h // 2

        # 在中心点绘制 step 信息
        cv2.putText(contour_image, f"Step {step_vi}", (center_x - 20, center_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)  # 黑色文本
    
    contour_image2 = cv2.resize(contour_image, (self.target_size * 2, self.target_size * 2))
    self.visualize_process("Debug View", contour_image2)