import cv2
import numpy as np


from .image_preprocessing import preprocess_image , light_detect
from .contour_detection import detect_contours, sort_quad, sort_quadrilaterals
from .color_detection import detect_colors, classify_color
from .detect_radio import detect_stretch_ratio
from .visualize_part import visualize_process , visualization_detect_contours
from .import_export_quad import import_quadrilaterals , export_quadrilaterals

time_start = 0  # 程序开始
time_end =0     # 程序结束

class ColorCodeDetector:
    def __init__(self, image_path,use_provided_quad=False, quad_file_path=None):
        # 初始化图像路径并加载图像
        self.image = cv2.imread(image_path)
        if self.image is None:
            raise ValueError("图像加载失败，请检查文件路径")
        self.quad_file_path = quad_file_path  # 边缘信息文件路径
        
        
        
        # 图片
        self.Sized_img = None  # 图像大小拉伸后
        self.closed_img = None  # 图像(闭运算)

        # 中间元素
        self.contours = []  # 轮廓
        self.quadrilaterals = []  # 有效外接四边形
        self.contours_ordered = []  # 排序后的外接四边形
        self.color_blocks = []  # 颜色块
        self.final_codes = []  # 颜色代码
        self.radio_stretch = 0.1  # 拉伸参数
        self.lightMax = 0 # 前百分之20%的亮度

        # 控制开关
        self.show_steps = True  # 控制是否显示处理步骤
        self.steps_fig = 0
        self.show_hsv = True  # False 则代表显示rgb
        self.use_provided_quad = use_provided_quad  # 是否使用传入的外接信息
        

        # 参数配置
        self.target_size_x = 400  # 标准处理尺寸 400
        self.target_size_y = 225 # 225
        self.min_contour_area = 50  # 最小轮廓面积 50
        self.min_screen_coef = 12  # 最小有效轮廓占总图像的1/x 8 
        self.max_screen_coef = 3  # 最小有效轮廓占总图像的1/x 3

        # 超参数
        self.HPsizeGaussian = 9  # 高斯模糊，默认5
        self.HPt1Canny = 50  # Canny阈值1，低于此值边缘被忽略，默认50
        self.HPt2Canny = 150  # Canny阈值2，高于此值边缘强边缘，默认150
        self.HPkernel = 9  # 形态学增强（闭运算核大小）
        self.HP_ts_radio = 0.3  # 拉伸突变容忍参数 
        self.HP_gamma = 0.7  # 指数映射比率 0.7
        self.HPbrightness_threshold = 120  # 亮度通道阈值，高于此值则加亮度
        self.HP_lightest_pencent= 20 #前百分之x的亮度，计算均值 20
        self.HP_lightest_Min_threshold = 20 # 亮度最低阈值20 
        self.HP_lightest_Max_threshold = 200 # 亮度最高阈值 170 猜的
        self.border_size = 50 # 补偿黑边长度
        self.center_factor = 2 / 3 # 取色区域，中心比率

        self.HP_Color_rules = [
            {"range": (0, 50), "name": "Red"},    # 0 <= h < 50 为红色
            {"range": (50, 70), "name": "Yellow"}, # 50 <= h < 70 为黄色
            {"range": (70, 150), "name": "Green"}, # 70 <= h < 150 为绿色
            {"range": (150, 283), "name": "Blue"}, # 150 <= h < 283 为蓝色
            {"range": (283, 360), "name": "Red"}  # 283 <= h < 360 为红色
        ]

        self.HP_Black_th= 0.3 # 黑色阈值


        # 绑定外部函数到类实例
        self.preprocess_image = preprocess_image.__get__(self)
        self.detect_contours = detect_contours.__get__(self)
        self.sort_quad = sort_quad.__get__(self)
        self.sort_quadrilaterals = sort_quadrilaterals.__get__(self)
        self.detect_colors = detect_colors.__get__(self)
        self.classify_color = classify_color.__get__(self)
        self.detect_stretch_ratio = detect_stretch_ratio.__get__(self)
        self.visualization_detect_contours = visualization_detect_contours.__get__(self)
        self.visualize_process = visualize_process.__get__(self)
        self.light_detect = light_detect.__get__(self)
        self.import_quadrilaterals = import_quadrilaterals.__get__(self)
        self.export_quadrilaterals = export_quadrilaterals.__get__(self)
        



    def analyze(self):
        """完整处理流程"""
        # 取消try；便于查询故障 try:
        # 先查亮度
        self.light_detect()
        if self.lightMax > self.HP_lightest_Max_threshold :
            # 亮度过量
            return{
                "Light_Max":self.lightMax,
                "Status":"Error",
                "Error_info":"LightTooBright",
            }
        elif self.lightMax < self.HP_lightest_Min_threshold:            
            return{
                "Light_Max":self.lightMax,
                "Status":"Error",
                "Error_info":"LightTooDim",
            }
        print("亮度：：：：")
        print(self.lightMax)
        cv2.waitKey()
     
        # 当引入外接的时候，使用这个逻辑
        if self.use_provided_quad :
            img = cv2.resize(self.image, (self.target_size_x, self.target_size_y), interpolation=cv2.INTER_AREA)
            self.visualize_process("Standard Size Pic", img)  # 标准尺寸图片
            self.import_quadrilaterals()
            self.detect_colors()
            self.visualization_detect_contours()
            cv2.waitKey()
            cv2.destroyAllWindows()
            return{
                "Status":"Success",
                "color_matrix": self.final_codes,
            }


        else:
            # 执行图像预处理、轮廓检测、颜色检测和可视化
            self.preprocess_image()  # 预处理 
            self.detect_contours()  # 轮廓检测 
            self.detect_stretch_ratio()  # 检测拉伸比率 
            
            self.detect_colors()  # 颜色检测

            self.visualization_detect_contours()  # 可视化找色块 


            cv2.waitKey()
            cv2.destroyAllWindows()
            
            
            return {
                "Status":"Success",
                "color_matrix": self.final_codes,
                "stretch_ratio": self.radio_stretch,
                
            }
    


import json
import time

if __name__ == "__main__":
    time_start = time.time()
    detector = ColorCodeDetector("./Sample/0331/Pic03_D1-SPEACIAL.png",\
                                 use_provided_quad=False,\
                                    quad_file_path="./ColorCodeDetector/testjson/123.json") # __init__
    result = detector.analyze()
    time_end = time.time()
    print(f"程序识别耗时： {time_end - time_start } ")

    if result.get('Status') == 'Success':
        print("识别结果：")
        for row in result.get('color_matrix', []):
            print(row)
        print("拉伸比例：")
        print(result.get('stretch_ratio'))
    else:
        print(result.get('Error_info'))
        # 打印错误信息
        print(f"亮度信息：{result.get('Light_Max')}")


