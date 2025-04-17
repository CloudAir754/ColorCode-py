import cv2
import os
from datetime import datetime
import json
from runPics import analyzeSingle
import time

def process_video(video_path):
    # 确保 out 文件夹存在
    
    # 计算时间
    start_time01 = time.time()

    out_folder = os.path.join(os.getcwd(), "out")
    if not os.path.exists(out_folder):
        os.makedirs(out_folder)

    # 获取当前时间并格式化为字符串，精确到毫秒
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")
    
    # 在 out 文件夹下创建以当前时间命名的子文件夹
    output_folder = os.path.join(out_folder, current_time)
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    cap = cv2.VideoCapture(video_path)
    # 打开视频文件
    frame_count = 0

    # 获取视频属性
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count_all = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Width: {width}, Height: {height}, FPS: {fps}, Frame Count: {frame_count_all}")

    # 计算一下并行计算需要的时间
    start_time02 = time.time()

    while cap.isOpened():
        ret, frame = cap.read()
        # 读取一帧，ret 为布尔值（是否成功读取），frame 为图像帧
        # frame 里面是纯粹的三维数组
        if not ret:
            break

        result = analyzeSingle(frame,False) # 如何存储和分析
        

        frame_count += 1

        # 在图片上绘制帧序号
        text = f"Frame: {frame_count}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1  # 字体大小
        font_color = (0, 255, 0)  # 字体颜色 (BGR 格式，绿色)
        thickness = 2  # 字体粗细
        position = (50, 50)  # 文本位置 (左上角)

        # 使用 cv2.putText 在图片上绘制文本
        cv2.putText(frame, text, position, font, font_scale, font_color, thickness)

        # 将图片保存到时间命名的子文件夹中
        frame_path = os.path.join(output_folder, f"frame_{frame_count}.jpg")
        cv2.imwrite(frame_path, frame)

    cap.release()
    end_time = time.time()
    print(f"从函数开始到函数结束，耗时：{end_time-start_time01}")   # 15.8
    print(f"批量图片处理（串行）耗时{end_time-start_time02}")      # 15.7


    return {"time":f"{end_time-start_time02}"}