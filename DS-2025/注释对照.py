# test03 代码

import cv2
import numpy as np

class ColorCodeDetector:
    def __init__(self, image_path):
        # 初始化类，读取图像
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
        """统一处理图像为三通道后再保存"""
        # 转换单通道图像为三通道
        if len(img.shape) == 2:
            display = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        else:
            display = img.copy()
        
        # 调整尺寸并添加标题
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

    # 透视变换
    def _warp_perspective(self, contour):
        """执行透视变换"""
        # 重新排列点：左上，右上，右下，左下
        pts = contour.reshape(4,2)
        rect = np.zeros((4,2), dtype=np.float32)
        
        # 通过和与差排序
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]  # 左上
        rect[2] = pts[np.argmax(s)]  # 右下
        
        d = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(d)]  # 右上
        rect[3] = pts[np.argmax(d)]  # 左下

        # 计算目标尺寸
        (tl, tr, br, bl) = rect
        width = max(np.linalg.norm(tr - tl), np.linalg.norm(br - bl))
        height = max(np.linalg.norm(bl - tl), np.linalg.norm(br - tr))
        
        # 目标点
        dst = np.array([
            [0, 0],
            [width-1, 0],
            [width-1, height-1],
            [0, height-1]], dtype=np.float32)

        # 透视变换
        M = cv2.getPerspectiveTransform(rect, dst)  # 计算透视变换矩阵
        warped = cv2.warpPerspective(self.orig, M, (int(width), int(height)))  # 执行透视变换
        return warped  # 返回变换后的图像

    # 颜色识别
    def _detect_colors(self, warped):
        """在3x3网格中检测颜色"""
        h, w = warped.shape[:2]  # 获取图像的高度和宽度
        cell_size = min(h, w) // 3  # 计算每个单元格的大小
        colors = []  # 存储检测到的颜色
        
        for row in range(3):
            for col in range(3):
                # 计算单元格坐标
                y1 = row * cell_size + cell_size//4
                y2 = y1 + cell_size//2
                x1 = col * cell_size + cell_size//4
                x2 = x1 + cell_size//2
                
                # 提取单元格区域
                cell = warped[y1:y2, x1:x2]
                
                # 转换为HSV颜色空间
                hsv = cv2.cvtColor(cell, cv2.COLOR_BGR2HSV)
                mean_hsv = np.mean(hsv, axis=(0,1))  # 计算HSV均值
                
                # 分类颜色
                colors.append(self._classify_color(mean_hsv))
        
        # 格式化为3x3矩阵
        return [colors[i*3:(i+1)*3] for i in range(3)]

    def _classify_color(self, hsv):
        """根据HSV值分类颜色"""
        hue, sat, val = hsv
        hue *= 2  # 转换为0-360范围
        
        if val < 50: return 'Black'  # 黑色
        if sat < 50 and val > 200: return 'White'  # 白色
        if (0 <= hue <= 15) or (165 <= hue <= 180): return 'Red'  # 红色
        if 40 <= hue <= 80: return 'Green'  # 绿色
        if 100 <= hue <= 140: return 'Blue'  # 蓝色
        return 'Unknown'  # 未知颜色

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
            
            # 步骤3: 透视变换
            warped = self._warp_perspective(largest)  # 执行透视变换
            
            self._add_step(warped, "6. Warped")  # 添加变换后的图像到处理步骤
            
            # 步骤4: 颜色检测
            #color_matrix = self._detect_colors(warped)  # 检测颜色
            color_matrix = self._detect_colors(self.orig)  # 检测颜色
            
            # 最终可视化
            result = np.hstack(self.process[:6] + [self.process[-1]])  # 拼接处理步骤图像
            # 上一行也有小问题问题
            cv2.imshow('Processing Steps', result)  # 显示处理步骤
            
            return {
                'status': 'success',  # 返回成功状态
                'colors': color_matrix,  # 返回颜色矩阵
                'warped': warped  # 返回变换后的图像
            }
            
        except Exception as e:
            # 创建错误显示图像
            err_img = np.zeros((400,400,3), dtype=np.uint8)
            cv2.putText(err_img, "3D Code Not Found", (50,200),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)
            self.process.append(err_img)  # 添加错误图像到处理步骤
            
            # 显示处理步骤时统一维度
            steps = [cv2.cvtColor(p, cv2.COLOR_GRAY2BGR) if len(p.shape)==2 else p 
                    for p in self.process[:5]]
            steps.append(self.process[-1])
            
            result = np.hstack(steps)  # 拼接处理步骤图像
            cv2.imshow('Processing Steps', result)  # 显示处理步骤
            
            return {
                'status': 'error',  # 返回失败状态
                'message': str(e)  # 返回错误信息
            }

if __name__ == "__main__":
    # 创建检测器实例并分析图像
    detector = ColorCodeDetector('./Sample/Pic2-2.jpg')
    result = detector.analyze()
    
    if result['status'] == 'success':
        print("Color Matrix:")
        for row in result['colors']:
            print(row)  # 打印颜色矩阵
        cv2.imshow('Warped Output', result['warped'])  # 显示变换后的图像
    
    cv2.waitKey(0)  # 等待用户按键
    cv2.destroyAllWindows()  # 关闭所有窗口