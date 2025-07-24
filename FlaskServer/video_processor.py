import cv2
import os,sys
from datetime import datetime
import time
import json
# 解决单独运行时模块导入问题

from ColorCodeDetector.ColorCodeDetector import ColorCodeDetector


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
        # 拉伸比率字典
        self.ratio = {
            self.STAGE_FULL_INFO: None,
            self.STAGE_BLUE_GONE: None,
            self.STAGE_RED_GONE: None
        }
        self.stability_threshold = stability_threshold
        self.current_stage_candidate = None
        self.candidate_streak = 0  # 当前候选状态连续出现的帧数

        self.frame_dic ={} # 帧号：数组
        self.step2_frameNum = 9999 # 第二阶段的最终编号
        self.step3_frameNum = 9999 # 第三阶段的最终编号
        self.min_frameNum = 0 # 最终搜索边界

        self.fps = 0 # 获取fps数据

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
        elif blue_count == 0 and red_count == 0:
            return self.STAGE_RED_GONE  # 第三阶段：红色+蓝色消失
        elif blue_count >0 :            # 第一阶段：只要有蓝色就行
            return self.STAGE_FULL_INFO
        else:
            return self.STAGE_TOO_BRIGHT  # 第零阶段：全信息；或者是啥也没有
    

    def process_frame(self, result, frame_info):
        """
        处理每一帧的结果b
            result: 一个json数组，包含当前帧的信息
            frame_info: 【字典】"frame_number" | "timestamp"        
        """
        current_candidate = self.determine_stage(result)

        # 先直接装入
        if current_candidate == self.STAGE_TOO_BRIGHT:
            # 这个状态意味着啥也没有
            return
        
        if self.stage_transitions[current_candidate] is None:
            # 未填入过信息
            self.stage_transitions[current_candidate] = {
            "color_matrix": result.get('color_matrix', []),
            "stretch_ratio": result.get('stretch_ratio'),
            "frame_info": frame_info
            }
        else:
            # 已经有了原始数据，但是补充未识别到的数据（黑色转彩色）
            Now_matrix = result.get('color_matrix', []) # 取出当前帧的颜色数组
            Old_matrix = self.stage_transitions[current_candidate]["color_matrix"] # 取出老的颜色数组
            # 遍历老颜色矩阵
            for i in range(3):
                for j in range(3):
                    current_color = Old_matrix[i][j]
                    # 如果当前位置是黑色或Zero，则进行处理；再看新数组是否有进步
                    if current_color in ['Black', 'Zero=Black']:
                        if Now_matrix[i][j] not in ['Black', 'Zero=Black']:
                            # 此时认为当前帧的效果更好（那应该是只进行增量替换呀；
                            #   哎呀无所谓~~，保不准替换了会乱，而且用增量替换难以回查原始数据）
                            self.stage_transitions[current_candidate] = {
                                "color_matrix": result.get('color_matrix', []),
                                "stretch_ratio": result.get('stretch_ratio'),
                                "frame_info": frame_info
                                }       
                
        #  这里调整逻辑；宏观分析，所有划入同一阶段的都进行处理；
        # 根据"color_matrix"数组，如果某位置识别到是除了黑色（"Black"，"Zero=Black"）的其他颜色，则进行替换   


    def _retry_with_adjusted_parameters(self,stage_to_search):
        """
        当stage_to_search阶段数据未检测到时，逐步降低参数阈值并重试
            返回 是否成功获得stage_to_search阶段(True)
        """

        original_brightness_threshold =  120 # ColorCodeDetector.HPbrightness_threshold
        original_gamma = 0.7 # ColorCodeDetector.HP_gamma
        
        # 尝试10次调整参数
        for attempt in range(1, 11):
            # 按比例降低参数
            reduction_factor = 0.85 ** attempt  # 每次降低20%
            new_min_threshold = max(15, original_brightness_threshold * reduction_factor)
            new_gamma = max(0.15, original_gamma * reduction_factor)
            
            print(f"Attempt {attempt}: Adjusting parameters - "
                  f"Min threshold: {new_min_threshold:.1f}, Gamma: {new_gamma:.2f}")
            
            # 创建新的ColorCodeDetector实例并处理关键帧 {帧号：数组}
            for frame_num, frame_dic1 in self.frame_dic.items():
                if self.min_frameNum !=9999 and frame_num > self.min_frameNum:
                    # 搜索边界有效 且 超限；则结束当前搜索
                    break
                                   
                # 创建临时检测器并调整参数
                detector = ColorCodeDetector(frame_dic1, pathSwtich=False)

                # 在改这里

                detector.HPbrightness_threshold = new_min_threshold
                detector.HP_gamma = new_gamma
                
                # 重新分析
                new_result = detector.analyze()
                
                if new_result.get('Status') == 'Success':
                    current_candidate = self.determine_stage(new_result)
                    if current_candidate == stage_to_search:
                        # 更新stage_transitions
              
                        # 设置当前帧信息(帧序号，秒数)
                        frame_info = {
                            "frame_number": frame_num,
                            "timestamp": frame_num / self.fps
                        }

                        self.stage_transitions[current_candidate] = {
                            "color_matrix": new_result.get('color_matrix', []),
                            "stretch_ratio": new_result.get('stretch_ratio'),
                            "frame_info": frame_info
                        }
                        
                        print(f"Successfully detected Stage 1 in attempt {attempt}")
                        return True
                    
        print("All attempts failed to detect Stage 1")
        return False 

    def store_all_frame(self,frame,frame_num):
        """存储帧，用于找不到第一阶段的情况"""
        self.frame_dic[frame_num]=frame

    def get_transition_info(self):
        """获取阶段转换信息，返回格式化的字符串"""
        # 原始信息打印保持不变
        print("="*50)
        print("The original information is as follows:")
        print(self.stage_transitions)
        print("="*50)
        
        # 检查第一阶段数据是否存在
        if self.stage_transitions[1] is None:
            print("Stage 1 content not detected (No Full)")
            
            # 先将相关帧号提取出来
            if self.stage_transitions[2] is not None:
                self.step2_frameNum = self.stage_transitions[2]['frame_info']['frame_number']
            if self.stage_transitions[3] is not None:
                self.step3_frameNum = self.stage_transitions[3]['frame_info']['frame_number']
            self.min_frameNum = min(self.step2_frameNum,self.step3_frameNum) # 最终搜索边界
            print(f"重新搜索边界为 0 ~ {self.min_frameNum}")

            # 尝试降低参数重新检测
            again_Succ = self._retry_with_adjusted_parameters(self.STAGE_FULL_INFO)
            
            if again_Succ is False :
                # 实在救不过来==》直接返回原始数据
                return self.stage_transitions
        
        # 获取第一阶段时间作为基准
        base_time = self.stage_transitions[1]['frame_info']['timestamp']
        
        # 转换姓名
        time_1_picMatrix = self.stage_transitions[1]["color_matrix"]
        name = self._convert_name(time_1_picMatrix)
        
        # 准备阶段信息
        stage_names = {
            1: "Full_Stage_1",
            2: "Blue_Gone_2", 
            3: "Red_Gone_3"
        }
        
        # 收集各阶段信息
        stage_details = []
        radio_details = ""

        for stage, data in self.stage_transitions.items():
            if data is not None:
                # 计算相对时间
                relative_time = round(data['frame_info']['timestamp'] - base_time, 3)
                absolute_time = round(data['frame_info']['timestamp'], 3)
                
                # 格式化拉伸比
                stretch_ratio = round(float(data["stretch_ratio"]), 3)
                self.ratio[stage] = stretch_ratio  # 将拉伸比计入字典

                radio_details += f"| {relative_time:1.2f} s \t | \t\t {stretch_ratio*100:3.2f}%\n"
                
                # 格式化颜色矩阵为3行
                color_matrix = "\n".join(
                    [f"    {str(row):<30}" for row in data['color_matrix']]
                )

                frame_num = data['frame_info']['frame_number']
                
                stage_details.append(
                    f"{stage_names.get(stage, f'Unknown stage {stage}')}:\n"
                    f"   Absolute time: {absolute_time:>7} s\n"
                    f"   Ori frame number: {frame_num} \n"
                    f"   Relative time: +{relative_time:>6} s\n"
                    f"   Stretching ratio:   {stretch_ratio:>7}\n"
                    f"   Color matrix:\n{color_matrix}"
                )
            else:
                stage_details.append(
                    f"{stage_names.get(stage, f'Unknown stage {stage}')}:\n"
                    f"  No content found at this stage"
                )    
        
        # 构建最终输出
        formatted_output = (
            f"\n{' TQR Code ':=^30}\n"
            f"Name: {name}\n"
            f"\nConversion information for each stage:\n"
            f"{'-'*30}\n"
            + "\n\n".join(stage_details) +
            f"\n{'-'*30}\n"
            f"{' End of analysis ':=^30}"
        )

        # 进行拉伸率评价
        pyhsical_condition = self._value_radio()

        phone_output = (
            "\n|==========="
            f"\n| Name: {name}\n"
            f"| Time and Strain:\n"
            f"{radio_details}"
            f"| Physical condition:\n"
            f"| {pyhsical_condition}"
            "\n|==========="
        )

        
        
        
        print("*"*60)
        print("The organized information is as follows:")
        print(formatted_output)
        print("*"*60)
        return phone_output

    def _value_radio(self):
        health_judeg = ""
        judge2,judge3 = 0
        try:
            # 先取出拉伸量
            data1 = self.ratio[self.STAGE_FULL_INFO]
            data2 = self.ratio[self.STAGE_BLUE_GONE]
            data3 = self.ratio[self.STAGE_RED_GONE]            
          
            if data2 > 1.3:
                # 第一阶段拉的快
                judge2 = 1
            if data3 > 2.3:
                judge3 = 1
            
            if judge2 and judge3:
                # 两次都拉伸的快
                health_judeg = "…HEALTHY…"
            else:
                health_judeg = "…UNHEALTHY…"

        except:
            health_judeg = "…HEALTHY…/UNHEATHY…"

        return health_judeg

    def _convert_name(self, pic_info):
        """
        由完全颜色（第一阶段）的图像，生成姓名        
        """
        # 1. 提取关键颜色（6个位置）
        key_positions = [
            (0, 1), (0, 2),  # 第一行的第2、3个元素
            (1, 0), (1, 2),  # 第二行的第1、3个元素
            (2, 0), (2, 1)   # 第三行的第1、2个元素
        ]
        key_colors = [pic_info[i][j] for i, j in key_positions]

        # 2. 将颜色转换为二进制码（Red=1, Blue=0）
        binary_code = [1 if color == 'Red' else 0 for color in key_colors]
        binary_str = ''.join(map(str, binary_code))  # 例如 "101010"

        # 3. 映射到预定义的64个人名（这里用简化的方式生成）
        name_index = int(binary_str, 2)  # 二进制转十进制（0-63）
        name_list = self._generate_name_list()  # 生成64个人名
        info_name = name_list[name_index]

        return info_name

    def _generate_name_list(self):
        """
        生成姓名组合       
        """
        # 生成2^6=64个人名（示例：用字母组合）
        first_names = ["Alice", "Bob", "Charlie", "David", "Eve", "Frank", "Grace", "Hank"]
        last_names = ["Smith", "Johnson", "Williams", "Brown", "Jones", "Miller", "Davis", "Wilson"]
        
        name_list = []
        for i in range(64):
            first = first_names[i % 8]
            last = last_names[i // 8]
            name_list.append(f"{first} {last}")
        
        return name_list


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

    print("The basic information of the video is as follows")
    print(f"Width: {width}, Height: {height}, FPS: {fps}, Frame Count: {frame_count}")
    turn_flag = False # 是否需要旋转
    if width > height:
        turn_flag = True
        print("--Need Turn--")
    lenth_time = frame_count / fps

    processor = VideoProcessor()  # 创建处理器实例

    processor.fps = fps # 传入fps数据

    frame_current = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # 添加旋转（假设所有视频都需要顺时针旋转90度）
        if turn_flag:
            frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        frame_current += 1
        
        # 分析当前帧 - 直接使用导入的 analyzeSingle

        result = analyzeSingle(frame, False)
        # result 是一个json数组，包含当前帧的信息
        
        frame_tmp = result.get("pic_toSave")
        # frame_tmp 这个的内容是一个合成图(左为定位，右为颜色注释)
        Ori_tmp = result.get("Ori_img")
        # Ori_tmp 这个的内容是一个原始图片（重整大小）

        # 设置当前帧信息(帧序号，秒数)
        frame_info = {
            "frame_number": frame_current,
            "timestamp": frame_current/fps
        }
        
        # 处理（当前帧）分析结果
        processor.process_frame(result, frame_info)
        # 存储当前帧
        processor.store_all_frame(frame=frame,frame_num=frame_current)


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
        frame_path2 = os.path.join(output_folder, f"Ori_{frame_current}.jpg")
        cv2.imwrite(frame_path, frame_tmp)
        cv2.imwrite(frame_path2, Ori_tmp)

    cap.release()
    video_info = processor.get_transition_info()
    return video_info,lenth_time

def analyzeSingle(PicPath,pathSwtich=True):
    """
    单张图片分析
        PicPath: 图片路径/图片数组
        pathSwtich: True路径 / False图片数组

    返回：
        "Status":"Success",
        "color_matrix": 3*3数组,
        "stretch_ratio": 拉伸比率,
        "Block_Counts": 块数量,
        "pic_toSave": 图片数组,
        "Ori_img": 原始图片（重整图片）
    """

    # 在视频处理流程，调用者使用的参数为False

    time_start = time.time()

    # v0.3 之后，只需要导入图片
    detector = ColorCodeDetector(PicPath,pathSwtich=pathSwtich) # __init__
    # False代表输入的是图片数组

    result = detector.analyze()
    time_end = time.time()
    # print(f"程序识别耗时： {time_end - time_start } ")
    # print(result.get('color_matrix', []))
    # print(result.get('status', []))

    return result


