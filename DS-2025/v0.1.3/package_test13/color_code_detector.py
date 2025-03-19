import cv2
import numpy as np
from image_preprocessing import preprocess_image
from contour_detection import detect_contours, sort_quad, sort_quadrilaterals
from color_detection import detect_colors, classify_color
from utils import detect_stretch_ratio, visualization_detect_contours

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
        self.quadrilaterals = []  # 有效内接四边形
        self.contours_ordered = []  # 排序后的内接四边形
        self.color_blocks = []  # 颜色块
        self.final_codes = []  # 颜色代码
        self.radio_stretch = 1.0  # 拉伸参数

        # 控制开关
        self.show_steps = True  # 控制是否显示处理步骤
        self.steps_fig = 0
        self.show_hsv = True  # False 则代表显示rgb

        # 参数配置
        self.target_size = 320  # 标准处理尺寸 320
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

        # 绑定外部函数到类实例
        self.preprocess_image = preprocess_image.__get__(self)
        self.detect_contours = detect_contours.__get__(self)
        self.sort_quad = sort_quad.__get__(self)
        self.sort_quadrilaterals = sort_quadrilaterals.__get__(self)
        self.detect_colors = detect_colors.__get__(self)
        self.classify_color = classify_color.__get__(self)
        self.detect_stretch_ratio = detect_stretch_ratio.__get__(self)
        self.visualization_detect_contours = visualization_detect_contours.__get__(self)

    def visualize_process(self, title_showd, img_showed):
        if self.show_steps:
            title_showd = "[Step " + str(self.steps_fig) + " ]= " + title_showd
            cv2.imshow(title_showd, img_showed)
            self.steps_fig += 1

    def analyze(self):
        """完整处理流程"""
        try:
            # 执行图像预处理、轮廓检测、颜色检测和可视化
            self.preprocess_image()  # 预处理
            self.detect_contours()  # 轮廓检测
            radio_stretch = self.detect_stretch_ratio()  # 检测拉伸比率
            print(f"平均拉伸比率: {radio_stretch:.2f}")

            self.detect_colors()  # 颜色检测

            color_matrix = self.final_codes

            self.visualization_detect_contours()  # 可视化找色块

            cv2.waitKey()
            cv2.destroyAllWindows()

            return {
                "color_matrix": color_matrix,
                "stretch_ratio": radio_stretch,
                "status": "success"
            }
        except Exception as e:
            return {
                "error": str(e),
                "status": "failed"
            }

if __name__ == "__main__":
    # 使用示例
    detector = ColorCodeDetector("./Sample/Pic00_1.jpg")
    result = detector.analyze()

    print("识别结果：")
    for row in result.get('color_matrix', []):
        print(row)

    # 当返回错误的时候，就报错
    if result['status'] == 'failed':
        print(f"识别失败: {result['error']}")