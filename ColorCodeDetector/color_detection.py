import cv2
import numpy as np

def detect_colors(self):
    """颜色检测主逻辑"""
    detectColor_image = cv2.resize(self.image.copy(), (self.target_size_x, self.target_size_y))

    # 在图片周围添加一圈黑边，确保文字不会超出范围
    border_size = self.border_size  # 可以根据需要调整边框大小
    detectColor_image = cv2.copyMakeBorder(detectColor_image, border_size, border_size, border_size, border_size, cv2.BORDER_CONSTANT, value=[0, 0, 0])



    color_means = []  # 存储每个区域的颜色平均值
    color_detects = []  # 存储颜色分类结果

    grid = self.grid 

    for row in range(3):  # 外层循环遍历行
        for col in range(3):  # 内层循环遍历列
            grid_use = grid[row][col]
            if grid_use is None:
                # print(f"空值：{row} {col}")
                color_means.append((0,0,0))
                color_detects.append("Zero=Black")
                continue
            
            # 如果是正常有效的区域        
            x, y, w, h = cv2.boundingRect(grid_use)
            
            x += border_size
            y += border_size

            center_factor = self.center_factor
            center_x = int(x + w * (0.5 - center_factor / 2))
            center_y = int(y + h * (0.5 - center_factor / 2))
            center_w = int(w * center_factor)
            center_h = int(h * center_factor)

            roi = detectColor_image[center_y:center_y + center_h, center_x:center_x + center_w]

        
            hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            h_mean = int(np.mean(hsv_roi[:, :, 0]))*2
            s_mean = int(np.mean(hsv_roi[:, :, 1]))/255
            v_mean = int(np.mean(hsv_roi[:, :, 2]))/255
            # 求 hsv 平均值
            # 且按照标准格式保存
            # 色调（H-360），饱和度（S-1），亮度（V-1）

            colors_group = (h_mean, s_mean, v_mean)
            color_means.append(colors_group)
            classify_color_text = classify_color(self, colors_group)
            color_detects.append(classify_color_text)
            # 按顺序添加颜色hsv列表

            # 展示颜色代码
            text_position = (center_x, center_y - 30)

            if self.show_details:

                # 显示两位小数
                cv2.putText(detectColor_image, f"H: {h_mean}", text_position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                cv2.putText(detectColor_image, f"S: {s_mean:.2f}", (center_x, center_y - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                cv2.putText(detectColor_image, f"V: {v_mean:.2f}", (center_x, center_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
                

            classify_text_position = (center_x, center_y + center_h + 20)
            cv2.putText(detectColor_image, classify_color_text, classify_text_position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            # 这一行是向图片添加颜色注释信息

            cv2.rectangle(detectColor_image, (center_x, center_y), (center_x + center_w, center_y + center_h), (0, 255, 0), 2)
            # 绘制颜色提取区域

    detectColor_image2 = cv2.resize(detectColor_image.copy(), (self.target_size_x * 2, self.target_size_y * 2)) # 放大已经被批注的图像
    self.visualize_process("Color Detect", detectColor_image2) 
    self.img_ColorDetect = detectColor_image2

    self.color_blocks = color_means
    self.final_codes = color_detects
    # 返回【颜色值列表和颜色分析结果列表】

    # 断点
    #cv2.waitKey()
    return

def classify_color(self, color): 
    """输入hsv序列，输出对应颜色字符"""
    
    h, s, v = color

    # 配置颜色分类的边界条件和颜色名称
    color_rules = self.HP_Color_rules

    # 黑色判定条件
    black_threshold = self.HP_Black_th
    if v < black_threshold:
        return "Black"

    # 根据配置列表判断颜色
    for rule in color_rules:
        lower, upper = rule["range"]
        if lower <= h < upper:
            return rule["name"]

    # 默认返回HSV值
    return f"H:{h},S:{s},V:{v}"

