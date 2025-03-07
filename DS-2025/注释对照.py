# 这个程序识别会导致只看一个小角落
# 以及机械性的分整张图为9块
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
        self.processed_img = None
        self.contours = []
        self.color_blocks = []
        self.final_codes = []
        
        # 参数配置
        self.target_size = 320       # 标准处理尺寸
        self.cell_tolerance = 15     # 网格排序容差
        self.min_contour_area = 50   # 最小轮廓面积
        
    def preprocess_image(self):
        """图像预处理流水线"""
        # 尺寸标准化
        img = cv2.resize(self.image, (self.target_size, self.target_size), 
                        interpolation=cv2.INTER_AREA)
        
        # 预处理流程：转换为灰度图像 -> 高斯模糊 -> Canny边缘检测
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5,5), 0)
        edged = cv2.Canny(blurred, 50, 150)
        
        # 形态学增强：闭运算以填充边缘检测后的孔洞
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
        closed = cv2.morphologyEx(edged, cv2.MORPH_CLOSE, kernel)
        
        # 保存处理后的图像
        self.processed_img = closed
        return closed
    
    def detect_contours(self):
        """增强的轮廓检测方法"""
        # 查找图像中的轮廓
        contours, _ = cv2.findContours(self.processed_img, 
                                      cv2.RETR_EXTERNAL, 
                                      cv2.CHAIN_APPROX_SIMPLE)
        
        # 动态尺寸计算：根据图像大小计算最小和最大轮廓尺寸
        img_size = self.target_size
        min_dim = img_size // 8
        max_dim = img_size // 3
        
        # 筛选符合条件的轮廓
        valid_contours = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < self.min_contour_area:
                continue
                
            x, y, w, h = cv2.boundingRect(cnt)
            if min_dim < w < max_dim and min_dim < h < max_dim:
                valid_contours.append(cnt)
        
        # 保存有效轮廓
        self.contours = valid_contours
        return valid_contours
    
    def perspective_transform(self):
        """透视变换校正"""
        # 寻找最大四边形轮廓
        max_area = 0
        best_quad = None
        for cnt in self.contours:
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02*peri, True)
            
            if len(approx) == 4 and cv2.isContourConvex(approx):
                area = cv2.contourArea(approx)
                if area > max_area:
                    max_area = area
                    best_quad = approx
        
        if best_quad is None:
            raise ValueError("未检测到有效的三维码区域")
        
        # 顶点排序
        src_points = self.order_points(best_quad.reshape(4,2))
        
        # 执行透视变换
        side_length = 300
        dst_points = np.array([[0,0], [side_length,0], 
                              [side_length,side_length], [0,side_length]], 
                             dtype=np.float32)
        
        # 计算透视变换矩阵并应用变换
        M = cv2.getPerspectiveTransform(src_points, dst_points)
        warped = cv2.warpPerspective(self.image, M, (side_length, side_length))
        
        return warped
    
    @staticmethod
    def order_points(pts):
        """改进的顶点排序算法"""
        # 按坐标和排序确定左上、右下
        sorted_by_sum = sorted(pts, key=lambda p: p[0]+p[1])
        tl, br = sorted_by_sum[0], sorted_by_sum[-1]
        
        # 按坐标差确定右上、左下
        remaining = [p for p in pts if not np.array_equal(p, tl) and not np.array_equal(p, br)]
        sorted_by_diff = sorted(remaining, key=lambda p: p[0]-p[1])
        tr, bl = sorted_by_diff[0], sorted_by_diff[-1]
        
        return np.array([tl, tr, br, bl], dtype=np.float32)
    
    def detect_colors(self, warped_img):
        """颜色检测主逻辑"""
        # 网格分割：将图像分为3x3的网格
        cell_size = warped_img.shape[0] // 3
        color_codes = []
        
        for row in range(3):
            for col in range(3):
                # 计算单元格区域
                x1 = col * cell_size + cell_size//4
                y1 = row * cell_size + cell_size//4
                x2 = x1 + cell_size//2
                y2 = y1 + cell_size//2
                
                # 提取单元格图像
                cell = warped_img[y1:y2, x1:x2]
                
                # 颜色分析：转换为HSV颜色空间并计算平均HSV值
                hsv = cv2.cvtColor(cell, cv2.COLOR_BGR2HSV)
                avg_hsv = np.mean(hsv, axis=(0,1))
                code = self.color_classification(avg_hsv)
                color_codes.append(code)
                
        # 按行列顺序重组颜色编码
        self.final_codes = [color_codes[i*3:(i+1)*3] for i in range(3)]
        return self.final_codes
    
    def color_classification(self, hsv_values):
        """增强的HSV颜色分类"""
        hue, sat, val = hsv_values
        hue = hue * 2  # OpenCV的Hue范围为0-179
        
        # 根据HSV值进行颜色分类
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
        # 复制图像用于显示
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
        
        # 显示结果图像
        cv2.imshow('Detection Result', display_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    def analyze(self):
        """完整处理流程"""
        try:
            # 执行图像预处理、轮廓检测、透视变换、颜色检测和可视化
            self.preprocess_image()
            self.detect_contours()
            warped = self.perspective_transform()
            color_matrix = self.detect_colors(warped)
            self.visualize_detection(warped)
            
            # 返回处理结果
            return {
                "color_matrix": color_matrix,
                "status": "success"
            }
        except Exception as e:
            # 捕获并返回错误信息
            return {
                "error": str(e),
                "status": "failed"
            }

if __name__ == "__main__":
    # 使用示例
    detector = ColorCodeDetector("./Sample/Pic2-2.jpg")
    result = detector.analyze()
    
    # 打印识别结果
    print("识别结果：")
    for row in result.get('color_matrix', []):
        print(row)
    
    # 如果识别失败，打印错误信息
    if result['status'] == 'failed':
        print(f"识别失败: {result['error']}")