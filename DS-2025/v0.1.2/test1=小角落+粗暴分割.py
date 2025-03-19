# 这个程序识别会导致只看一个小角落
# 以及机械性的分整张图为9块
# 所有的TODO替换为to1do
import cv2
import numpy as np
from sklearn.cluster import KMeans

class ColorCodeDetector:
    def __init__(self, image_path):
        # 初始化图像路径并加载图像
        self.image = cv2.imread(image_path)
        if self.image is None:
            raise ValueError("图像加载失败，请检查文件路径")
        
        # 初始化处理后的图像、轮廓、颜色块和最终颜色编码
        self.processed_img = None # 图像
        self.contours = []  # 轮廓
        self.color_blocks = [] # 颜色快
        self.final_codes = [] # 颜色代码
        
        # 参数配置
        self.target_size = 320       # 标准处理尺寸
        # self.cell_tolerance = 15     # (未使用)网格排序容差
        
        self.min_contour_area = 50   # 最小轮廓面积
        
    def preprocess_image(self):
        """图像预处理流水线"""
        # 尺寸标准化
        img = cv2.resize(self.image, (self.target_size, self.target_size), 
                        interpolation=cv2.INTER_AREA)
        # interpolation=cv2.INTER_AREA 区域插值，适合缩小图像
        # 预处理流程
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5,5), 0)
        edged = cv2.Canny(blurred, 50, 150) # Canny边缘检测


        ''''
        cv2.Canny(image, threshold1, threshold2)
        第一个阈值，用于边缘检测中的滞后阈值处理。低于此阈值的边缘会被忽略。
        第二个阈值，用于边缘检测中的滞后阈值处理。高于此阈值的边缘会被认为是强边缘。
        '''

        # 形态学增强：闭运算以填充边缘检测后的孔洞
        # 原来的核为5*5，改为9*9是为了让部分残缺的方框补全
        # TO1DO 后期代码，若有问题，检查这里是否闭合
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9,9))
        closed = cv2.morphologyEx(edged, cv2.MORPH_CLOSE, kernel)
        
        # cv2.imshow("closed",closed)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        #closed 存放的实际是闭环的轮廓信息

        self.processed_img = closed
        return closed
    
    def detect_contours(self):
        """增强的轮廓检测方法"""
        contours, _ = cv2.findContours(self.processed_img, 
                                      cv2.RETR_EXTERNAL, 
                                      cv2.CHAIN_APPROX_SIMPLE)
        # img==>非零像素被视为前景（白色），零像素被视为背景（黑色）

        # 动态尺寸计算
        img_size = self.target_size
        min_dim = img_size // 8
        max_dim = img_size // 3
        
        # 筛选符合条件的轮廓
        valid_contours = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < self.min_contour_area:
                # 这个最小轮廓面积也可以这么算……
                # 跳过噪音（特别小的轮廓）
                continue
                
            x, y, w, h = cv2.boundingRect(cnt) 
            # 上面这个函数用于计算最小外接矩形
            if min_dim < w < max_dim and min_dim < h < max_dim:
                valid_contours.append(cnt)
        
        self.contours = valid_contours
        # 存储合法的框框

        # 添加的代码——显示有效框框（部分情况下显示不全；
        #   则需要回看cloese元素是否成功不全）
        # 在原图像上绘制边缘（轮廓）
        # 创建一个原图像的副本，避免修改原图
        contour_image = cv2.resize(self.image, (self.target_size, self.target_size))  # 调整大小以匹配处理后的图像
        # 绘制所有检测到的轮廓
        cv2.drawContours(contour_image, valid_contours, -1, (0, 255, 0), 2)  # 绿色轮廓，线宽为2

        # 显示叠加轮廓后的图像
        cv2.imshow("Contours on Original Image", contour_image)
        cv2.waitKey(0)  # 等待用户按键
        cv2.destroyAllWindows()  # 关闭窗口




        return valid_contours
    
    def perspective_transform(self):
        """透视变换校正"""
        
        # 寻找最大四边形
        max_area = 0
        best_quad = None
        for cnt in self.contours:
            # 对所有合法边缘进行操作：
            peri = cv2.arcLength(cnt, True) 
            # peri=轮廓周长==>True 代表轮廓闭合
            approx = cv2.approxPolyDP(cnt, 0.02*peri, True)
            # 对轮廓进行多边形逼近，
            # 将轮廓近似为一个更简单的多边形；
            # 0.02*peri 是逼近精度
            # 返回值approx为一个多边形轮廓Numpy数组
           

            if len(approx) == 4 and cv2.isContourConvex(approx):
                # 检查“is四边形”&&“凸多边形”
                area = cv2.contourArea(approx)
                if area > max_area:
                    max_area = area
                    best_quad = approx
        
        if best_quad is None:
            raise ValueError("未检测到有效的三维码区域")
        
        # 每一个approx，都是存了四个点坐标（4，1，2）
        # 顶点排序；传入面积最大的点进行排序
        
        src_points = self.order_points(best_quad.reshape(4,2))
        # TO1DO 注意，此处返回的点的顺序可能不是预期的
        # 实际的顺序是【右下、左下、左上、右上】

        # 执行透视变换
        side_length = 300 
        dst_points = np.array([[0,0], [side_length,0], 
                              [side_length,side_length], [0,side_length]], 
                             dtype=np.float32)
        # TO1DO 这里又是按照【左上、右上、右下、左下】的顺序进行排布
        #目标正方形顶点坐标？300？

        M = cv2.getPerspectiveTransform(src_points, dst_points)
        # 计算透视变换矩阵 M，将源四边形的顶点映射到目标正方形的顶点。
        warped = cv2.warpPerspective(self.image, M, (side_length, side_length))
        # 应用透视变换，将原图像校正为 300x300 的正方形。
        
        return warped
    
    @staticmethod
    def order_points(pts):
        """
        为什么使用静态方法？（哦~和c差不多）
        order_points 方法的逻辑完全依赖于传入的参数 
        pts（四边形的四个顶点坐标），
        而不需要访问类的属性或实例的属性。
        静态方法的特点就是不需要 self 或 cls 参数，
        因此非常适合这种场景。        
        """

        # 传入一个四边形的四个点坐标
        """改进的顶点排序算法"""
        # 按坐标和排序确定左上、右下
        sorted_by_sum = sorted(pts, key=lambda p: p[0]+p[1])
        # 这个lambda表达式真的绝了……
        # 意思是按照x+y的值进行排序
        tl, br = sorted_by_sum[0], sorted_by_sum[-1]
        # 0 是最小的点，左上？；-1 是最大的点，右下？
        # TO1DO 这一句明显反了啊
        
        # 按坐标差确定右上、左下
        remaining = [p for p in pts if not np.array_equal(p, tl) and not np.array_equal(p, br)]
        sorted_by_diff = sorted(remaining, key=lambda p: p[0]-p[1])
        tr, bl = sorted_by_diff[0], sorted_by_diff[-1]
        
        print(np.array([tl, tr, br, bl], dtype=np.float32))# ?
        
        # 左上、右上、右下、左下?
        # TO1DO 这一句明显反了啊
        # 实际的顺序是【右下、左下、左上、右上】

        return np.array([tl, tr, br, bl], dtype=np.float32)
    
    def detect_colors(self, warped_img):
        """颜色检测主逻辑"""
        
        # cv2.imshow("warped_img",warped_img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        # 网格分割
        cell_size = warped_img.shape[0] // 3
        # 这个网格分割也是有问题的
        # 逻辑问题；即，不应当按照图片3*3分割
        #   而是按照划到的边缘块进行分割
        
        color_codes = []
        
        for row in range(3):
            for col in range(3):
                # 按照长宽比划为9块
                # 计算单元格区域
                x1 = col * cell_size + cell_size//4
                y1 = row * cell_size + cell_size//4
                x2 = x1 + cell_size//2
                y2 = y1 + cell_size//2
                
                cell = warped_img[y1:y2, x1:x2]
                
                # 颜色分析
                """"
                HSV 将颜色信息分解为三个独立的分量：
                    ​H (Hue, 色调)：表示颜色的类型（如红色、绿色、蓝色等）。
                    ​S (Saturation, 饱和度)：表示颜色的纯度（饱和度越高，颜色越鲜艳）。
                    ​V (Value, 亮度)：表示颜色的亮度。
                
                """
                hsv = cv2.cvtColor(cell, cv2.COLOR_BGR2HSV)
                # 将图像从bgr颜色空间切换到hsv颜色空间
                avg_hsv = np.mean(hsv, axis=(0,1))
                # 计算 HSV 图像的平均值
                # 指定在高度（axis=0）和宽度（axis=1）两个维度上计算平均值。

                code = self.color_classification(avg_hsv)
                # 映射
                color_codes.append(code)
                
        # 按行列顺序重组
        self.final_codes = [color_codes[i*3:(i+1)*3] for i in range(3)]
        return self.final_codes
    
    def color_classification(self, hsv_values):
        """增强的HSV颜色分类"""
        hue, sat, val = hsv_values
        hue = hue * 2  # OpenCV的Hue范围为0-179
        
        if val < 40:
            return '11'  # 黑色
        
        if sat < 50 and val > 200:
            return '00'  # 白色
        
        if (0 <= hue <= 15) or (165 <= hue <= 180):
            return '01'  # 红色
        
        if 45 <= hue <= 75:
            return '10'  # 绿色
        
        if 90 <= hue <= 130: 
            return '11'  # 蓝色
        
        return '00'  # 默认返回白色
    
    def visualize_detection(self, warped_img):
        """可视化检测结果"""
        display_img = warped_img.copy()
        cell_size = display_img.shape[0] // 3
        
        # 绘制网格线
        for i in range(1,3):
            cv2.line(display_img, (i*cell_size,0), (i*cell_size,display_img.shape[0]), (0,255,0), 2)
            cv2.line(display_img, (0,i*cell_size), (display_img.shape[1],i*cell_size), (0,255,0), 2)
        
        # 显示颜色编码
        for row in range(3):
            for col in range(3):
                x = col * cell_size + 20
                y = row * cell_size + 40
                code = self.final_codes[row][col]
                cv2.putText(display_img, code, (x,y), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
        
        cv2.imshow('Detection Result', display_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    def analyze(self):
        """完整处理流程"""
        try:
            # 执行图像预处理、轮廓检测、透视变换、颜色检测和可视化
            
            self.preprocess_image() # 预处理
            self.detect_contours() # 轮廓检测
            warped = self.perspective_transform() # 透视变换?
            # TO1DO 透视矫正有点顺序问题
            # 矫正完成后的图片就已经只能看到小角落了()
            
            color_matrix = self.detect_colors(warped) # 颜色检测
            self.visualize_detection(warped) # 可视化检测结果
            return {
                "color_matrix": color_matrix,
                "status": "success"
            }
        except Exception as e:
            return {
                "error": str(e),
                "status": "failed"
            }

if __name__ == "__main__":
    # 使用示例
    detector = ColorCodeDetector("./Sample/Pic2-2.jpg")
    result = detector.analyze()
    
    print("识别结果：")
    # 从 result的json格式文件取内容
    # 取出键'color_matrix'的值
    for row in result.get('color_matrix', []):
        print(row)
        
    # 当返回错误的时候，就报错
    if result['status'] == 'failed':
        print(f"识别失败: {result['error']}")