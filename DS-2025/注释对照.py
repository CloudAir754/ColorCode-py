# test 07
import cv2
import numpy as np

class ColorCodeDetector:
    def __init__(self, image_path):
        # 读取图像
        self.orig = cv2.imread(image_path)
        if self.orig is None:
            raise ValueError("Image load failed")  # 如果图像加载失败，抛出异常
        
        # 用于存储处理步骤的图像
        self.process = []  
        # 目标图像大小
        self.target_size = 800
        # 最小轮廓面积
        self.min_contour_area = 500

    # 可视化控制
    def _add_step(self, img, title):
        """将处理步骤的图像保存到列表中"""
        # 如果图像是单通道，转换为三通道
        if len(img.shape) == 2:
            display = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        else:
            display = img.copy()
        
        # 调整图像大小并添加标题
        display = cv2.resize(display, (400, 400))
        cv2.putText(display, title, (10,30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
        self.process.append(display)  # 将处理步骤图像添加到列表中

    # 图像预处理
    def _preprocess(self):
        """图像预处理流程"""
        # 调整图像大小
        img = cv2.resize(self.orig, (self.target_size, self.target_size))
        self._add_step(img, "1. Original")  # 添加原始图像到处理步骤
        
        # 灰度化
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        self._add_step(gray, "2. Grayscale")  # 添加灰度图像到处理步骤
        
        # 直方图均衡化
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        equalized = clahe.apply(gray)
        self._add_step(equalized, "3. Equalized")  # 添加均衡化图像到处理步骤
        
        # 边缘检测
        blurred = cv2.GaussianBlur(equalized, (5,5), 0)  # 高斯模糊，减少噪声
        edged = cv2.Canny(blurred, 50, 150)  # Canny边缘检测
        self._add_step(edged, "4. Canny Edges")  # 添加边缘检测图像到处理步骤
        
        # 形态学操作
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))  # 创建5x5矩形核
        closed = cv2.morphologyEx(edged, cv2.MORPH_CLOSE, kernel)  # 形态学闭合操作，填充边缘间隙
        self._add_step(closed, "5. Morphology")  # 添加形态学操作图像到处理步骤
        
        return closed  # 返回处理后的二值图像

    # 颜色识别
    def _detect_colors(self, img):
        """在3x3网格中检测颜色"""
        h, w = img.shape[:2]  # 获取图像的高度和宽度
        cell_size = min(h, w) // 3  # 计算每个单元格的大小
        colors = []  # 存储检测到的颜色
        
        # 创建副本用于绘制注释
        annotated_image = img.copy()
        
        for row in range(3):
            for col in range(3):
                # 计算单元格坐标
                y1 = row * cell_size + cell_size//4
                y2 = y1 + cell_size//2
                x1 = col * cell_size + cell_size//4
                x2 = x1 + cell_size//2
                
                # 提取单元格区域
                cell = img[y1:y2, x1:x2]
                
                # 获取单元格中心点
                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2
                
                # 分类颜色
                color = self._classify_color(cell)  # 返回对应区域的颜色
                colors.append(color)
                
                # 在图像上标注颜色
                cv2.putText(annotated_image, str(color), (center_x - 20, center_y), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                
                # 在颜色检测区域绘制矩形
                cv2.rectangle(annotated_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # 格式化为3x3矩阵
        return [colors[i*3:(i+1)*3] for i in range(3)], annotated_image

    def _classify_color(self, cell):
        """根据BGR值分类颜色"""
        # 计算单元格的平均颜色
        mean_color = np.mean(cell, axis=(0,1))
        b, g, r = mean_color
        
        # 简单的颜色分类
        if r > 200 and g < 50 and b < 50: return 'Red'  # 红色
        if g > 200 and r < 50 and b < 50: return 'Green'  # 绿色
        if b > 200 and r < 50 and g < 50: return 'Blue'  # 蓝色
        if r > 200 and g > 200 and b > 200: return 'White'  # 白色
        if r < 50 and g < 50 and b < 50: return 'Black'  # 黑色
        
        # 如果颜色未知，返回RGB值
        return f"({int(r)}, {int(g)}, {int(b)})"

    # 主流程
    def analyze(self):
        """主分析流程"""
        try:
            # 步骤1: 预处理
            processed = self._preprocess()
            
            # 步骤2: 查找轮廓
            contours, _ = cv2.findContours(processed,
                                          cv2.RETR_EXTERNAL,
                                          cv2.CHAIN_APPROX_SIMPLE)
            
            # 过滤轮廓
            candidates = []
            for cnt in contours:
                area = cv2.contourArea(cnt)  # 计算轮廓面积
                if area < self.min_contour_area: continue  # 跳过面积过小的轮廓
                
                # 近似多边形
                peri = cv2.arcLength(cnt, True)  # 计算轮廓周长
                approx = cv2.approxPolyDP(cnt, 0.02*peri, True)  # 多边形近似
                if len(approx) == 4:  # 如果是四边形
                    candidates.append(approx)  # 添加到候选列表
            
            if not candidates:
                raise ValueError("No 3D code found")  # 如果没有候选，抛出异常
            
            # 选择最大轮廓
            largest = max(candidates, key=cv2.contourArea)  # 选择面积最大的轮廓
            
            # 步骤3: 颜色检测
            color_matrix, annotated_image = self._detect_colors(self.orig)  # 检测颜色
            self._add_step(annotated_image, "6. Annotated Colors")  # 添加标注图像到处理步骤
            
            # 显示每个处理步骤
            for i, step in enumerate(self.process):
                cv2.imshow(f'Step {i+1}', step)
            
            return {
                'status': 'success',  # 返回成功状态
                'colors': color_matrix,  # 返回颜色矩阵
                'annotated': annotated_image  # 返回标注图像
            }
            
        except Exception as e:
            # 创建错误显示图像
            err_img = np.zeros((400,400,3), dtype=np.uint8)
            cv2.putText(err_img, "3D Code Not Found", (50,200),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)
            self.process.append(err_img)  # 添加错误图像到处理步骤
            
            # 显示每个处理步骤
            for i, step in enumerate(self.process):
                cv2.imshow(f'Step {i+1}', step)
            
            return {
                'status': 'error',  # 返回失败状态
                'message': str(e)  # 返回错误信息
            }

if __name__ == "__main__":
    # 创建检测器实例并分析图像
    detector = ColorCodeDetector('./Sample/Pic2.jpg')
    result = detector.analyze()
    
    if result['status'] == 'success':
        print("Color Matrix:")
        for row in result['colors']:
            print(row)  # 打印颜色矩阵
        cv2.imshow('Annotated Colors', result['annotated'])  # 显示标注图像
    
    cv2.waitKey(0)  # 等待用户按键
    cv2.destroyAllWindows()  # 关闭所有窗口