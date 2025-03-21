import cv2
import os
from datetime import datetime

def process_video(video_path):
    # 确保 out 文件夹存在
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
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        # 将图片保存到时间命名的子文件夹中
        frame_path = os.path.join(output_folder, f"frame_{frame_count}.jpg")
        cv2.imwrite(frame_path, frame)

    cap.release()