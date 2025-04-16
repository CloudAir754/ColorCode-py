import cv2
import numpy as np

# TODO 比率检测函数也需要改
def detect_stretch_ratio(self):
    """检测拉伸比率，并检查波动是否过大。"""
    if not self.contours_ordered:
        self.Status += "No valid contours_ordered [detec_radio.py]" 
        if self.show_steps:
            print("没有检测到有效的四边形，无法计算拉伸比率。")
        return 
    
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
            Error_info = (f"第 {i+1} 个四边形的拉伸比率 {ratio:.2f} \
                          与均值 {mean_ratio:.2f} 的差异超过阈值 \
                            {self.HP_ts_radio*100:.0f}%。")# 太长了……
            self.Status += Error_info
            if self.show_steps:
                print(Error_info)

    
    self.radio_stretch = mean_ratio
    return 

