import cv2
import numpy as np

def detect_colors(self):
    """颜色检测主逻辑"""
    detectColor_image = cv2.resize(self.image.copy(), (self.target_size, self.target_size))

    color_means = []  # 存储每个区域的颜色平均值
    color_detects = []  # 存储颜色分类结果

    for quad in self.quadrilaterals:
        x, y, w, h = cv2.boundingRect(quad)

        center_factor = 2 / 3
        center_x = int(x + w * (0.5 - center_factor / 2))
        center_y = int(y + h * (0.5 - center_factor / 2))
        center_w = int(w * center_factor)
        center_h = int(h * center_factor)

        roi = detectColor_image[center_y:center_y + center_h, center_x:center_x + center_w]

        if self.show_hsv:
            hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            h_mean = int(np.mean(hsv_roi[:, :, 0]))
            s_mean = int(np.mean(hsv_roi[:, :, 1]))
            v_mean = int(np.mean(hsv_roi[:, :, 2]))
            colors_group = (h_mean, s_mean, v_mean)
            color_means.append(colors_group)
            classify_color_text = classify_color(self, colors_group)
            color_detects.append(classify_color_text)

            text_position = (center_x, center_y - 30)
            cv2.putText(detectColor_image, f"H: {h_mean}", text_position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(detectColor_image, f"S: {s_mean}", (center_x, center_y - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(detectColor_image, f"V: {v_mean}", (center_x, center_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        else:
            b_mean = int(np.mean(roi[:, :, 0]))
            g_mean = int(np.mean(roi[:, :, 1]))
            r_mean = int(np.mean(roi[:, :, 2]))
            colors_group = (b_mean, g_mean, r_mean)
            color_means.append(colors_group)

            classify_color_text = classify_color(self, colors_group)
            color_detects.append(classify_color_text)

            hex_color = "#{:02X}{:02X}{:02X}".format(r_mean, g_mean, b_mean)
            text_position = (center_x, center_y - 10)
            cv2.putText(detectColor_image, hex_color, text_position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        classify_text_position = (center_x, center_y + center_h + 20)
        cv2.putText(detectColor_image, classify_color_text, classify_text_position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        cv2.rectangle(detectColor_image, (center_x, center_y), (center_x + center_w, center_y + center_h), (0, 255, 0), 2)

    detectColor_image2 = cv2.resize(detectColor_image.copy(), (self.target_size * 2, self.target_size * 2))
    self.visualize_process("Color Detect", detectColor_image2)

    self.color_blocks = color_means
    self.final_codes = color_detects

    return color_means

def classify_color(self, color):
    """根据颜色代码判断颜色类别（红色、蓝色、绿色、黑色）。"""
    if self.show_hsv:
        h, s, v = color
        h = h * 2
        From0 = 0
        From0_endRed = 60
        FromRed_endGreen = 165
        FromGreen_endBlue = 300
        FromBlue_end360 = 360
        Black_below = 60

        if v < Black_below:
            return "Black"

        if From0 <= h < From0_endRed:
            return "Red"
        elif h < FromRed_endGreen:
            return "Green"
        elif h < FromGreen_endBlue:
            return "Blue"
        elif h < FromBlue_end360:
            return "Red"

        return f"H:{h},S:{s},V:{v}"
    else:
        b, g, r = color
        return f"B:{b},G:{g},R:{r}"