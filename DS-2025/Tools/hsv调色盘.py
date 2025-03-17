import cv2
import numpy as np
import random

# 创建一个空图像
def create_blank_image():
    return np.zeros((200, 400, 3), np.uint8)

# 更新图像颜色
def update_image_color(image, hsv):
    h, s, v = hsv
    image[:] = (h, s, v)
    bgr = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)
    return bgr

# 空函数，用于滑动条的回调
def nothing(x):
    pass

# 随机化指定通道的值
def randomize_channels(hsv, random_h, random_s, random_v):
    h, s, v = hsv
    if random_h:
        h = random.randint(0, 179)
    if random_s:
        s = random.randint(0, 255)
    if random_v:
        v = random.randint(0, 255)
    return h, s, v

# 创建窗口
cv2.namedWindow('HSV Color Picker')
cv2.namedWindow('Color Display')  # 新窗口用于显示颜色

# 创建滑动条
cv2.createTrackbar('H', 'HSV Color Picker', 0, 179, nothing)
cv2.createTrackbar('S', 'HSV Color Picker', 0, 255, nothing)
cv2.createTrackbar('V', 'HSV Color Picker', 0, 255, nothing)

# 创建复选框（是否对当前通道随机化）
random_h = False
random_s = False
random_v = False
cv2.createTrackbar('Random H', 'HSV Color Picker', 0, 1, nothing)
cv2.createTrackbar('Random S', 'HSV Color Picker', 0, 1, nothing)
cv2.createTrackbar('Random V', 'HSV Color Picker', 0, 1, nothing)

# 创建按钮（通过滑动条模拟）
cv2.createTrackbar('Randomize', 'HSV Color Picker', 0, 1, nothing)

# 初始化图像
image = create_blank_image()

# 随机化状态
randomize_active = False

while True:
    # 获取滑动条的当前值
    h = cv2.getTrackbarPos('H', 'HSV Color Picker')
    s = cv2.getTrackbarPos('S', 'HSV Color Picker')
    v = cv2.getTrackbarPos('V', 'HSV Color Picker')
    
    # 获取复选框的状态
    random_h = cv2.getTrackbarPos('Random H', 'HSV Color Picker')
    random_s = cv2.getTrackbarPos('Random S', 'HSV Color Picker')
    random_v = cv2.getTrackbarPos('Random V', 'HSV Color Picker')
    
    # 获取按钮的状态
    randomize_state = cv2.getTrackbarPos('Randomize', 'HSV Color Picker')
    if randomize_state == 1:
        randomize_active = True
    else:
        randomize_active = False
    
    # 如果随机化激活，随机化指定通道的值
    if randomize_active:
        h, s, v = randomize_channels((h, s, v), random_h, random_s, random_v)
        cv2.setTrackbarPos('H', 'HSV Color Picker', h)
        cv2.setTrackbarPos('S', 'HSV Color Picker', s)
        cv2.setTrackbarPos('V', 'HSV Color Picker', v)
        # 添加 0.05 秒的延迟
        cv2.waitKey(5)
    
    # 更新图像颜色
    hsv_color = np.array([h, s, v])
    bgr_color = update_image_color(image, hsv_color)
    
    # 显示颜色在另一个窗口中
    color_display = np.zeros((200, 200, 3), np.uint8)
    color_display[:] = bgr_color[0, 0]  # 将颜色填充到整个窗口
    cv2.imshow('Color Display', color_display)
    
    # 显示控制窗口
    cv2.imshow('HSV Color Picker', bgr_color)
    
    # 按下ESC键退出
    if cv2.waitKey(1) == 27:
        break

cv2.destroyAllWindows()