import cv2
import numpy as np

def preprocess_image(self):
    """图像预处理流水线，规范大小，找出边缘（初步）"""
    # 尺寸标准化
    img = cv2.resize(self.image, (self.target_size, self.target_size), interpolation=cv2.INTER_AREA)
    self.Sized_img = img

    # 图像增强
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    v_channel = hsv[:, :, 2]  # 提取 V 通道

    # 对亮度较高的像素进行指数映射
    v_normalized = v_channel / 255.0  # 归一化到 [0, 1]
    brightness_threshold = self.HPbrightness_threshold  # 最大的一个分量
    gamma = self.HP_gamma  # <0 就是放大
    mask = v_channel > brightness_threshold  # 创建高亮区域的掩码
    v_enhanced = np.where(mask, np.power(v_normalized, gamma) * 255, v_channel)  # 仅增强高亮区域
    v_enhanced = v_enhanced.astype(np.uint8)  # 恢复像素范围

    # 将增强后的 V 通道合并回 HSV 图像
    hsv[:, :, 2] = v_enhanced
    img_enhanced = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)  # 转换回 BGR 颜色空间

    gray = cv2.cvtColor(img_enhanced, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (self.HPsizeGaussian, self.HPsizeGaussian), 0)  # 高斯模糊
    edged = cv2.Canny(blurred, self.HPt1Canny, self.HPt2Canny)  # Canny边缘检测



    # 形态学增强：闭运算以填充边缘检测后的孔洞
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (self.HPkernel, self.HPkernel))  # 返回一个卷积核
    closed = cv2.morphologyEx(edged, cv2.MORPH_CLOSE, kernel)

    self.visualize_process("Standard Size Pic", img)  # 标准尺寸图片
    self.visualize_process("Gamma Func.", img_enhanced)  # 增亮图像
    self.visualize_process("Gray Func.", gray)  # 灰度图像
    self.visualize_process("GaussianBlur Func.", blurred)  # 高斯模糊图像
    self.visualize_process("Canny Edge Func.", edged)  # Canny 边缘检测
    self.visualize_process("CLOSED Func.", closed)

    self.closed_img = closed
    return 