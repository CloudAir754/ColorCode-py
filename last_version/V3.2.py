import cv2
import numpy as np

# 读取图片
image_path = 'Pic2-2.jpg'
image = cv2.imread(image_path)

def ColorDetect(color):
    R, G, B = 0, 0, 0
    if color[0] > 100:
        R = 1
    if color[1] > 100:
        G = 1
    if color[2] > 100:
        B = 1
    print(f"RGB: {color}, Detected: R={R}, G={G}, B={B}")
    if R == 1 and G == 0 and B == 0:
        return '01'
    if R == 0 and G == 1 and B == 1:
        return '00'
    if R == 1 and G == 1 and B == 0:
        return '10'
    return "11"

# 检查图像是否成功读取
if image is None:
    print("图像读取失败")
else:
    # 调整图片大小
    height, width = image.shape[:2]
    image = cv2.resize(image, (320, 320), interpolation=cv2.INTER_AREA)

    # 转换为灰度图像
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 高斯模糊
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)

    # 边缘检测
    edged = cv2.Canny(blurred, 20, 80)

    # 形态学变换
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    closed = cv2.morphologyEx(edged, cv2.MORPH_CLOSE, kernel)

    # cv2.imshow("KERNEL", kernel)
    cv2.imshow("CLOS", closed)

    # 查找轮廓
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 存储颜色块信息和长宽比分类
    colors = []
    aspect_ratios = []

    # 遍历每个轮廓
    for contour in contours:
        # 计算轮廓的边界框
        x, y, w, h = cv2.boundingRect(contour)

        # 计算长宽比
        aspect_ratio = w / h if h != 0 else 0
        aspect_ratios.append(aspect_ratio)

        # 筛选符合颜色块特征的轮廓
        if 40 < w < 100 and 40 < h < 100:  # 假设颜色块的宽高范围为20到100(示例图像适合40-100)
            # 提取当前轮廓区域
            grid = image[int(y + h / 4):int(y + h * 3 / 4), int(x + w / 4):int(x + w * 3 / 4)]  # 截取中间区域

            # 计算网格的平均颜色
            avg_color = cv2.mean(grid)[:3]
            avg_color = (int(avg_color[2]), int(avg_color[1]), int(avg_color[0]))  # 转换为RGB格式

            # 存储颜色信息
            colors.append((x, y, w, h, avg_color))

            # 绘制边界框（可选）
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # 选择过半的长宽比分类
    ratio_class = 1 if sum(1 for r in aspect_ratios if r <= 1.5) > len(aspect_ratios) / 2 else 2
    print(f"过半的长宽比分类为: {ratio_class}")

    # 按位置排序（从上到下，从左到右）
    # 按位置排序（从左到右，从上到下），加入容差判断行
    tolerance = 20
    colors.sort(key=lambda c: (round(c[1] / tolerance), c[0]))  # 按列排序，每列再从上到下  # 按行排序，每行再从左到右

    # 输出排序后的颜色块坐标信息
    for idx, color in enumerate(colors):
        print(f"排序后颜色块 {idx + 1} 的坐标: (x={color[0]}, y={color[1]})")

    color_image = []  # 颜色信息记录

    # 输出每个颜色块的颜色信息（从左往右，从上往下）
    for idx, color in enumerate(colors):
        x, y, w, h, avg_color = color
        print(f"颜色块 {idx + 1} (位置: ({x}, {y}), 尺寸: ({w}, {h})): {avg_color}")
        colorback_ = ColorDetect(avg_color)
        color_image.append(colorback_)

    # 显示检测结果
    cv2.imshow('Detected Color Blocks', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # 输出最终颜色编码
    print("Color is :")
    for index, color in enumerate(colors):
        x, y, _, _, _ = color
        print(f"颜色块 {index + 1} 的坐标: (x={x}, y={y})")
    print(color_image)
    for index, item in enumerate(color_image):
        print(f"{index}: {item}")
