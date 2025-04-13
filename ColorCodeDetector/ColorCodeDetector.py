import cv2
import numpy as np


from .image_preprocessing import preprocess_image
from .contour_detection import detect_contours, sort_quad
from .color_detection import detect_colors, classify_color
from .detect_radio import detect_stretch_ratio
from .visualize_part import visualize_process , visualization_detect_contours
from .diagonal_postion import detect_green_diagonals,locate_nine

time_start = 0  # 程序开始
time_end =0     # 程序结束

class ColorCodeDetector:
    def __init__(self, image_path):
        # 初始化图像路径并加载图像
        self.image = cv2.imread(image_path)
        if self.image is None:
            raise ValueError("图像加载失败，请检查文件路径")
                 
        # 图片
        self.Sized_img = None  # 图像大小拉伸后
        self.closed_img = None  # 图像(闭运算)

        # 中间元素
        self.contours = []  # 轮廓 
        self.contours_ordered = [] # 内部有效的轮廓集
        self.quadrilaterals = []  # 有效外接四边形

        self.green_diagonals = []  # 存储绿色对角线色块的数组(x,y,w,h)
        self.grid = [[None for _ in range(3)] for _ in range(3)] # 按照行列索引外接四边形

        self.color_blocks = []  # 颜色块
        self.final_codes = []  # 颜色代码
        self.radio_stretch = 0.1  # 拉伸参数

        # 控制开关
        self.show_steps = True  # 控制是否显示处理步骤
        self.steps_fig = 0
        self.show_details = False # 展示hsv

        # 参数配置
        self.target_size_x = 225  # 标准处理尺寸 225
        self.target_size_y = 400 # 400
        self.min_contour_area = 200  # 最小轮廓面积 200 是 正常拉伸的经验值
        self.min_screen_coef = 15  # 最小有效轮廓占总图像的1/x 12 
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
            {"range": (0, 50), "name": "Red"},    # 0 <= h < 50 为红色！！
            {"range": (50, 70), "name": "Yellow"}, # 50 <= h < 70 为黄色
            {"range": (70, 150), "name": "Green"}, # 70 <= h < 150 为绿色！！
            {"range": (150, 283), "name": "Blue"}, # 150 <= h < 283 为蓝色！！
            {"range": (283, 360), "name": "Red"}  # 283 <= h < 360 为红色！！
        ]

        self.HP_Black_th= 0.65 # 黑色阈值 0.3


        # 绑定外部函数到类实例
        self.preprocess_image = preprocess_image.__get__(self)
        self.detect_contours = detect_contours.__get__(self)
        self.sort_quad = sort_quad.__get__(self)
        self.detect_colors = detect_colors.__get__(self)
        self.classify_color = classify_color.__get__(self)
        self.detect_stretch_ratio = detect_stretch_ratio.__get__(self)
        self.visualization_detect_contours = visualization_detect_contours.__get__(self)
        self.visualize_process = visualize_process.__get__(self)    

        
        self.detect_green_diagonals = detect_green_diagonals.__get__(self)
        self.locate_nine = locate_nine.__get__(self)
        



    def analyze(self):
        """完整处理流程"""    
        # 执行图像预处理、轮廓检测、颜色检测和可视化
        self.preprocess_image()  # 预处理 
        self.detect_contours()  # 轮廓检测 
        self.detect_stretch_ratio()  # 检测拉伸比率 
        
        # 新增的绿色对角线检测
        self.detect_green_diagonals()
        self.locate_nine()

        self.detect_colors()  # 颜色检测
        self.visualization_detect_contours()  # 可视化找色块 


        cv2.waitKey()
        cv2.destroyAllWindows()
        
        
        return {
            "Status":"Success",
            "color_matrix": self.final_codes,
            "stretch_ratio": self.radio_stretch,
            
        }
    