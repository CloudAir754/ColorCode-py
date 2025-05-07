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
    
    # 定义阶段常量
    STAGE_UNINITIALIZED = -1
    STAGE_TOO_BRIGHT = 0
    STAGE_FULL_INFO = 1
    STAGE_BLUE_GONE = 2
    STAGE_RED_GONE = 3

    def __init__(self,stability_threshold=5):
        """
        初始化视频处理器
        :param stability_threshold: 状态稳定所需的连续帧数
        """
        self.stage = self.STAGE_UNINITIALIZED
        self.stage_transitions = {
            self.STAGE_FULL_INFO: None,
            self.STAGE_BLUE_GONE: None,
            self.STAGE_RED_GONE: None
        }
        self.stability_threshold = stability_threshold
        self.current_stage_candidate = None
        self.candidate_streak = 0  # 当前候选状态连续出现的帧数

    def determine_stage(self, result):
        """
        根据分析结果确定当前阶段
            result 是每个帧的识别信息json数组
        """
        if result.get('Status') != 'Success':
            return self.STAGE_TOO_BRIGHT  # 第零阶段：太亮；或者太亮。反正是无效信息
        
        color_matrix = result.get('color_matrix', [])
        red_count = sum(row.count("Red") for row in color_matrix)
        blue_count = sum(row.count("Blue") for row in color_matrix)
        
        if blue_count == 0 and red_count > 0:
            return self.STAGE_BLUE_GONE  # 第二阶段：蓝色消失
        elif red_count == 0:
            return self.STAGE_RED_GONE  # 第三阶段：红色消失
        elif red_count+blue_count == 6:#(9-3=6;最标准的一阶段)
            return self.STAGE_FULL_INFO
        else:
            return self.STAGE_TOO_BRIGHT  # 第一阶段：全信息；或者是啥也没有
    

    def process_frame(self, result, frame_info):
        """
        处理每一帧的结果b
            result: 一个json数组，包含当前帧的信息
            frame_info: 【字典】"frame_number" | "timestamp"        
        """
        current_candidate = self.determine_stage(result)
        
        # 如果当前候选状态与之前不同，重置计数器
        if current_candidate != self.current_stage_candidate:
            self.current_stage_candidate = current_candidate
            self.candidate_streak = 1
        else:
            self.candidate_streak += 1

        # 仅当候选状态连续出现足够帧数，并且是下一个合法状态时，才更新阶段
        if (
            self.candidate_streak >= self.stability_threshold
            and current_candidate > self.stage  # 确保状态是递进的
        ):
            self.stage = current_candidate
            # 记录阶段转换信息（如果是第一次进入该阶段）
            if current_candidate in self.stage_transitions and self.stage_transitions[current_candidate] is None:
                self.stage_transitions[current_candidate] = {
                    "color_matrix": result.get('color_matrix', []),
                    "stretch_ratio": result.get('stretch_ratio'),
                    "frame_info": frame_info
                }


    def get_transition_info(self):
        """获取阶段转换信息"""
        return self.stage_transitions

def process_video(video_path):
    """
    处理视频的主函数（被route文件调用）
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

    processor = VideoProcessor()  # 创建处理器实例

    frame_current = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # 添加旋转（假设所有视频都需要顺时针旋转90度）
        frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        frame_current += 1
        
        # 分析当前帧 - 直接使用导入的 analyzeSingle

        result = analyzeSingle(frame, False)
        # result 是一个json数组，包含当前帧的信息
        
        frame_tmp = result.get("pic_toSave")

        # 设置当前帧信息(帧序号，秒数)
        frame_info = {
            "frame_number": frame_current,
            "timestamp": frame_current/fps
        }
        
        # 处理分析结果
        processor.process_frame(result, frame_info)
        
        # 在图片上绘制帧序号

        text = f"Frame: {frame_info}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1
        font_color = (0, 255, 0)
        thickness = 2
        position = (50, 50)
        cv2.putText(frame_tmp, text, position, font, font_scale, font_color, thickness)

        # 将图片保存到时间命名的子文件夹中
        frame_path = os.path.join(output_folder, f"frame_{frame_current}.jpg")
        cv2.imwrite(frame_path, frame_tmp)

    cap.release()
    video_info = processor.get_transition_info()
    return video_info,lenth_time


if __name__ == "__main__":
    # 测试视频路径
    test_video = "./Sample/trailer.mp4"  # 替换为实际测试视频路径
    
    if not os.path.exists(test_video):
        print(f"测试视频文件 {test_video} 不存在")
    else:
        print("开始处理视频...")
        video_info, lenth_time = process_video(test_video)
        print("\n处理结果:")
        print(f"视频信息：{video_info}")
        print(f"视频长度：{lenth_time}")

