# test02 代码
import cv2
import numpy as np

class VisualCodeDetector:
    def __init__(self, image_path):
        # 初始化类，读取图像
        self.image = cv2.imread(image_path)
        if self.image is None:
            raise ValueError("图像加载失败")  # 如果图像加载失败，抛出异常
        
        # 控制是否显示处理步骤
        self.show_steps = True
        # 存储检测到的轮廓
        self.contours = []
        # 存储最终选择的四边形
        self.final_quad = None
        
        # 初始化处理参数
        self.target_size = 800  # 目标图像大小
        self.min_contour_area = 100  # 最小轮廓面积
        self.max_contour_area = 100000  # 最大轮廓面积

    def visualize(self, image, title="Step"):
        """通用可视化方法，显示处理步骤的图像"""
        if self.show_steps:
            # 调整图像大小并显示
            cv2.imshow(title, cv2.resize(image, (600, 600)))
            cv2.waitKey(100)  # 等待100ms，方便观察

    def preprocess(self):
        """带可视化的预处理流程"""
        # 调整尺寸
        h, w = self.image.shape[:2]  # 获取图像的高度和宽度
        scale = self.target_size / max(h, w)  # 计算缩放比例
        # TODO 这个缩放比例的逻辑不对啊，不应该是按照
        self.image = cv2.resize(self.image, None, fx=scale, fy=scale)  # 按比例缩放图像
        self.visualize(self.image, "0.原始图像")  # 显示原始图像

        # 灰度化
        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)  # 将图像转换为灰度图
        self.visualize(gray, "1.灰度化")  # 显示灰度图像

        # 直方图均衡
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))  # 创建CLAHE对象
        equalized = clahe.apply(gray)  # 应用直方图均衡化
        self.visualize(equalized, "2.直方图均衡")  # 显示均衡化后的图像

        # 边缘检测
        blurred = cv2.GaussianBlur(equalized, (5,5), 0)  # 高斯模糊，减少噪声
        edged = cv2.Canny(blurred, 50, 150)  # Canny边缘检测
        self.visualize(edged, "3.Canny边缘检测")  # 显示边缘检测结果

        # 形态学操作
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))  # 创建5x5矩形核
        closed = cv2.morphologyEx(edged, cv2.MORPH_CLOSE, kernel)  # 形态学闭合操作，填充边缘间隙
        self.visualize(closed, "4.形态学闭合")  # 显示形态学操作结果

        return closed  # 返回处理后的二值图像

    def find_candidates(self, processed):
        """带可视化的候选区域查找"""
        # 查找轮廓
        contours, _ = cv2.findContours(processed, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        
        # 绘制所有轮廓
        contour_img = self.image.copy()  # 复制原始图像
        cv2.drawContours(contour_img, contours, -1, (0,255,0), 2)  # 绘制所有轮廓
        self.visualize(contour_img, "5.所有检测到的轮廓")  # 显示绘制结果

        # 筛选候选轮廓
        candidates = []
        for cnt in contours:
            area = cv2.contourArea(cnt)  # 计算轮廓面积
            if not (self.min_contour_area < area < self.max_contour_area):
                continue  # 跳过不符合面积要求的轮廓

            peri = cv2.arcLength(cnt, True)  # 计算轮廓周长
            approx = cv2.approxPolyDP(cnt, 0.02*peri, True)  # 多边形近似
            
            if len(approx) == 4 and cv2.isContourConvex(approx):  # 如果是凸四边形
                candidates.append(approx)  # 添加到候选列表
                
        # 绘制候选四边形
        candidate_img = self.image.copy()
        cv2.drawContours(candidate_img, candidates, -1, (0,0,255), 3)  # 绘制候选四边形
        self.visualize(candidate_img, "6.候选四边形")  # 显示候选四边形

        return candidates  # 返回候选四边形列表

    def select_best_quad(self, candidates):
        """可视化选择最佳四边形"""
        if not candidates:
            self.show_failure("未找到候选四边形")  # 如果没有候选，显示失败信息
            raise ValueError("未找到有效四边形")  # 抛出异常

        # 按面积排序
        candidates.sort(key=cv2.contourArea, reverse=True)  # 按面积从大到小排序
        
        # 显示前3个候选
        for i, cnt in enumerate(candidates[:3]):
            quad_img = self.image.copy()
            cv2.drawContours(quad_img, [cnt], -1, (0,255,255), 3)  # 绘制候选四边形
            self.visualize(quad_img, f"7.候选{i+1}")  # 显示候选四边形

        # 选择最大且符合颜色特征的
        self.final_quad = candidates[0]  # 选择面积最大的候选
        final_img = self.image.copy()
        cv2.drawContours(final_img, [self.final_quad], -1, (255,0,0), 3)  # 绘制最终选择的四边形
        self.visualize(final_img, "8.最终选择四边形")  # 显示最终选择的四边形
        return self.final_quad  # 返回最终选择的四边形

    def show_failure(self, message):
        """显示失败信息"""
        fail_img = self.image.copy()  # 复制原始图像
        cv2.putText(fail_img, message, (50,100), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)  # 在图像上绘制失败信息
        self.visualize(fail_img, "ERROR")  # 显示失败信息
        cv2.waitKey(0)  # 等待用户按键

    def analyze(self):
        try:
            # 处理流程
            processed = self.preprocess()  # 预处理图像
            candidates = self.find_candidates(processed)  # 查找候选区域
            best_quad = self.select_best_quad(candidates)  # 选择最佳四边形
            
            # 透视变换和颜色识别（此处省略具体实现）
            return {"status": "success"}  # 返回成功状态
            
        except Exception as e:
            if self.show_steps:
                cv2.destroyAllWindows()  # 关闭所有窗口
            return {"status": "failed", "error": str(e)}  # 返回失败状态和错误信息

if __name__ == "__main__":
    try:
        detector = VisualCodeDetector("./Sample/Pic2-2.jpg")  # 创建检测器实例
        result = detector.analyze()  # 分析图像
        print(result)  # 打印结果
        cv2.waitKey(0)  # 等待用户按键
        cv2.destroyAllWindows()  # 关闭所有窗口
    except Exception as e:
        print(f"初始化错误: {str(e)}")  # 打印初始化错误信息