import cv2
import os,sys
from datetime import datetime
import time
import json
# 解决单独运行时模块导入问题
try:
    from runPics import analyzeSingle
except ImportError:
    # 如果是单独运行，尝试添加项目根目录到系统路径
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.append(project_root)
    from runPics import analyzeSingle

class VideoProcessor:
    def __init__(self):
        self.stage = -1  # 当前阶段：-1=未初始化, 0=太亮, 1=全信息, 2=蓝色消失, 3=红色消失
        self.prev_stage = -1
        self.stage_transitions = {
            1: None,  # 第一阶段开始的信息
            2: None,  # 第二阶段开始的信息
            3: None   # 第三阶段开始的信息
        }
        self.prev_color_matrix = None
        self.prev_stretch_ratio = None
    
    def determine_stage(self, result):
        """根据分析结果确定当前阶段"""
        if result.get('Status') != 'Success':
            return 0  # 第零阶段：太亮
        
        color_matrix = result.get('color_matrix', [])
        red_count = sum(row.count("Red") for row in color_matrix)
        blue_count = sum(row.count("Blue") for row in color_matrix)
        
        if blue_count == 0 and red_count > 0:
            return 2  # 第二阶段：蓝色消失
        elif red_count == 0:
            return 3  # 第三阶段：红色消失
        else:
            return 1  # 第一阶段：全信息
    
    def process_frame(self, result, frame_info):
        """处理每一帧的结果"""
        current_stage = self.determine_stage(result)
        
        # 如果阶段发生变化，记录转换信息
        if current_stage != self.stage:
            self.prev_stage = self.stage
            self.stage = current_stage
            
            # 记录阶段转换时的信息
            if current_stage in [1, 2, 3] and self.stage_transitions[current_stage] is None:
                # 使用前一帧的信息（如果有），或者当前帧的信息
                if self.prev_color_matrix is not None and self.prev_stretch_ratio is not None:
                    self.stage_transitions[current_stage] = {
                        "color_matrix": self.prev_color_matrix,
                        "stretch_ratio": self.prev_stretch_ratio,
                        "frame_info": frame_info
                    }
                elif result.get('Status') == 'Success':
                    self.stage_transitions[current_stage] = {
                        "color_matrix": result.get('color_matrix', []),
                        "stretch_ratio": result.get('stretch_ratio'),
                        "frame_info": frame_info
                    }
        
        # 保存当前帧信息供下一帧使用
        if result.get('Status') == 'Success':
            self.prev_color_matrix = result.get('color_matrix', [])
            self.prev_stretch_ratio = result.get('stretch_ratio')
        else:
            self.prev_color_matrix = None
            self.prev_stretch_ratio = None
    
    def get_transition_info(self):
        """获取阶段转换信息"""
        return self.stage_transitions

def process_video(video_path):
    """
    处理视频的主函数
    :param video_path: 视频文件路径
    :return: 处理结果字段video_info,原始视频长度lenth_time
    """
    
    video_info = "这个是视频信息，占位"

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


    # 获取视频属性
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Width: {width}, Height: {height}, FPS: {fps}, Frame Count: {frame_count}")
    lenth_time = frame_count / fps


    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        
        # 分析当前帧 - 直接使用导入的 analyzeSingle
        result = analyzeSingle(frame, False)
        
        # 设置当前帧信息
        frame_info = {
            "frame_number": frame_count,
            "timestamp": time.time() - start_time
        }
        
        # 处理分析结果
        processor.process_frame(result, frame_info)
        
        # 在图片上绘制帧序号
        text = f"Frame: {frame_count}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1
        font_color = (0, 255, 0)
        thickness = 2
        position = (50, 50)
        cv2.putText(frame, text, position, font, font_scale, font_color, thickness)

        # 将图片保存到时间命名的子文件夹中
        frame_path = os.path.join(output_folder, f"frame_{frame_count}.jpg")
        cv2.imwrite(frame_path, frame)

    cap.release()

    return video_info,lenth_time


if __name__ == "__main__":
    # 测试视频路径
    test_video = "./Sample/trailer.mp4"  # 替换为实际测试视频路径
    
    if not os.path.exists(test_video):
        print(f"测试视频文件 {test_video} 不存在")
    else:
        print("开始处理视频...")
        result = process_video(test_video)
        print("\n处理结果:")
        print(json.dumps(result, indent=2))

