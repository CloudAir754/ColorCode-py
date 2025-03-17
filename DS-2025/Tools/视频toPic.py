import cv2
import os

def split_video_to_frames(video_path):
    # 获取视频文件名（不含路径和扩展名）
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    
    # 生成输出文件夹路径
    output_folder = os.path.join(os.path.dirname(video_path), video_name)
    
    # 打开视频文件
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print("Error: Could not open video.")
        return
    
    # 确保输出文件夹存在
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    frame_count = 0
    
    while True:
        # 读取一帧
        ret, frame = cap.read()
        
        # 如果读取失败，退出循环
        if not ret:
            break
        
        # 保存帧到文件
        frame_filename = os.path.join(output_folder, f"frame_{frame_count:04d}.jpg")
        cv2.imwrite(frame_filename, frame)
        
        frame_count += 1
    
    # 释放视频对象
    cap.release()
    print(f"Finished splitting video into {frame_count} frames. Frames saved in: {output_folder}")

if __name__ == "__main__":
    # 视频文件路径
    video_path = "./Sample/videos/aaa4=5s.mp4"
    
    # 调用函数拆分视频帧
    split_video_to_frames(video_path)