import cv2
import numpy as np
from image_preprocessing import preprocess_image , light_detect
from contour_detection import detect_contours, sort_quad, sort_quadrilaterals
from color_detection import detect_colors, classify_color
from detect_radio import detect_stretch_ratio
from visualize_part import visualize_process , visualization_detect_contours

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
        self.quadrilaterals = []  # 有效外接四边形
        self.contours_ordered = []  # 排序后的外接四边形
        self.color_blocks = []  # 颜色块
        self.final_codes = []  # 颜色代码
        self.radio_stretch = 1.0  # 拉伸参数
        self.lightMax = 0 # 前百分之20%的亮度

        # 控制开关
        self.show_steps = True  # 控制是否显示处理步骤
        self.steps_fig = 0
        self.show_hsv = True  # False 则代表显示rgb

        # 参数配置
        self.target_size_x = 400  # 标准处理尺寸 320
        self.target_size_y = 300
        self.min_contour_area = 50  # 最小轮廓面积 50
        self.min_screen_coef = 12  # 最小有效轮廓占总图像的1/x 8 
        self.max_screen_coef = 3  # 最小有效轮廓占总图像的1/x 3

        # 超参数
        self.HPsizeGaussian = 5  # 高斯模糊，默认5
        self.HPt1Canny = 50  # Canny阈值1，低于此值边缘被忽略，默认50
        self.HPt2Canny = 150  # Canny阈值2，高于此值边缘强边缘，默认150
        self.HPkernel = 9  # 形态学增强（闭运算核大小）
        self.HP_ts_radio = 0.3  # 拉伸突变容忍参数 
        self.HP_gamma = 0.7  # 指数映射比率
        self.HPbrightness_threshold = 120  # 亮度通道阈值，高于此值则加亮度
        self.HP_lightest_pencent= 20 #前百分之x的亮度，计算均值
        self.HP_lightest_Min_threshold = 20 # 亮度最低阈值20 
        self.HP_lightest_Max_threshold = 170 # 亮度最高阈值 170 猜的

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



    def analyze(self):
        """完整处理流程"""
        # 取消try；便于查询故障 try:
        # 先查亮度
        self.light_detect()
        if self.lightMax > self.HP_lightest_Max_threshold or \
            self.lightMax < self.HP_lightest_Min_threshold:
            print("亮度不合格，不再处理")
            print(self.lightMax)
            return{
                "Error_info":"LightError",
            }

        cv2.waitKey()
        # 执行图像预处理、轮廓检测、颜色检测和可视化
        self.preprocess_image()  # 预处理 
        self.detect_contours()  # 轮廓检测 
        self.detect_stretch_ratio()  # 检测拉伸比率 
        
        self.detect_colors()  # 颜色检测

        self.visualization_detect_contours()  # 可视化找色块 

        cv2.waitKey()
        cv2.destroyAllWindows()

        return {
            "color_matrix": self.final_codes,
            "stretch_ratio": self.radio_stretch,
            
        }

if __name__ == "__main__":
    # 使用示例
    detector = ColorCodeDetector("./Sample/Pic01_1.png") # __init__
    result = detector.analyze()

    if result.get('Error_info'):
        print(result.get('Error_info'))

    print("识别结果：")
    for row in result.get('color_matrix', []):
        print(row)
    print("拉伸比例：")
    print(result.get('stretch_ratio'))