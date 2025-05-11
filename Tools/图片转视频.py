import cv2
import os
from typing import List

def images_to_video(
    image_paths: List[str],
    output_video_path: str,
    fps: float = 24.0,
    duration_per_image: float = 2.0,
    resolution: tuple = (1920, 1080)
) -> None:
    """
    将多张图片按顺序组合成视频
    
    参数:
        image_paths: 图片路径列表
        output_video_path: 输出视频路径
        fps: 视频帧率
        duration_per_image: 每张图片显示的持续时间(秒)
        resolution: 视频分辨率 (宽度, 高度)
    """
    # 检查图片列表是否为空
    if not image_paths:
        print("错误: 没有提供图片路径")
        return
    
    # 检查所有图片是否存在
    for img_path in image_paths:
        if not os.path.exists(img_path):
            print(f"错误: 图片 {img_path} 不存在")
            return
    
    # 创建视频写入对象
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 也可以使用 'avc1' 或其他编码器
    video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, resolution)
    
    # 计算每张图片需要写入的帧数
    frames_per_image = int(fps * duration_per_image)
    
    try:
        for img_path in image_paths:
            # 读取图片
            img = cv2.imread(img_path)
            if img is None:
                print(f"警告: 无法读取图片 {img_path}, 跳过")
                continue
            
            # 调整图片大小到目标分辨率
            img_resized = cv2.resize(img, resolution)
            
            # 将图片写入视频多次以达到持续时间
            for _ in range(frames_per_image):
                video_writer.write(img_resized)
                
        print(f"视频已成功创建: {output_video_path}")
        
    except Exception as e:
        print(f"创建视频时出错: {e}")
        
    finally:
        # 释放视频写入对象
        video_writer.release()


if __name__ == "__main__":
    # 示例用法
    import glob
    
    # 获取当前目录下所有jpg图片
    image_files = sorted(glob.glob("Sample/Sample-01/*.png"))
    
    if not image_files:
        print("当前目录中没有找到jpg图片")
    else:
        # 自定义参数
        output_video = "Sample/output_video.mp4"
        frames_per_second = 30.0
        image_duration = 1.5  # 每张图片显示3秒
        video_resolution = (1080, 1920)  # 720p分辨率
        
        print(f"将以下图片转换为视频: {image_files}")
        images_to_video(
            image_paths=image_files,
            output_video_path=output_video,
            fps=frames_per_second,
            duration_per_image=image_duration,
            resolution=video_resolution
        )