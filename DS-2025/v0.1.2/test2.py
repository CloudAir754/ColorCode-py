# 边缘检测有大问题
# 功能缺少！

import cv2
import numpy as np

class VisualCodeDetector:
    def __init__(self, image_path):
        self.image = cv2.imread(image_path)
        if self.image is None:
            raise ValueError("图像加载失败")
        
        self.show_steps = True  # 控制是否显示处理步骤
        self.contours = []
        self.final_quad = None
        
        # 初始化处理参数
        self.target_size = 800
        self.min_contour_area = 100
        self.max_contour_area = 100000

    def visualize(self, image, title="Step"):
        """通用可视化方法"""
        if self.show_steps:
            cv2.imshow(title, cv2.resize(image, (600, 600)))
            cv2.waitKey(100)

    def preprocess(self):
        """带可视化的预处理流程"""
        # 调整尺寸
        h, w = self.image.shape[:2]
        # TODO 这个缩放比例的逻辑不对啊，整体算的时候，应该是按照原图尺寸去匹配边缘
        scale = self.target_size / max(h, w)
        self.image = cv2.resize(self.image, None, fx=scale, fy=scale)
        self.visualize(self.image, "0.原始图像")

        # 灰度化
        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        self.visualize(gray, "1.灰度化")

        # 直方图均衡
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        equalized = clahe.apply(gray)
        self.visualize(equalized, "2.直方图均衡")

        # 边缘检测
        blurred = cv2.GaussianBlur(equalized, (5,5), 0)
        edged = cv2.Canny(blurred, 50, 150)
        self.visualize(edged, "3.Canny边缘检测")

        # 形态学操作
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
        closed = cv2.morphologyEx(edged, cv2.MORPH_CLOSE, kernel)
        self.visualize(closed, "4.形态学闭合")

        return closed

    def find_candidates(self, processed):
        """带可视化的候选区域查找"""
        # 查找轮廓
        contours, _ = cv2.findContours(processed, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        
        # 绘制所有轮廓
        contour_img = self.image.copy()
        cv2.drawContours(contour_img, contours, -1, (0,255,0), 2)
        self.visualize(contour_img, "5.所有检测到的轮廓")

        # 筛选候选轮廓
        candidates = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if not (self.min_contour_area < area < self.max_contour_area):
                continue

            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02*peri, True)
            
            if len(approx) == 4 and cv2.isContourConvex(approx):
                candidates.append(approx)
                
        # 绘制候选四边形
        candidate_img = self.image.copy()
        cv2.drawContours(candidate_img, candidates, -1, (0,0,255), 3)
        self.visualize(candidate_img, "6.候选四边形")

        return candidates

    def select_best_quad(self, candidates):
        """可视化选择最佳四边形"""
        if not candidates:
            self.show_failure("未找到候选四边形")
            raise ValueError("未找到有效四边形")

        # 按面积排序
        candidates.sort(key=cv2.contourArea, reverse=True)
        
        # 显示前3个候选
        for i, cnt in enumerate(candidates[:3]):
            quad_img = self.image.copy()
            cv2.drawContours(quad_img, [cnt], -1, (0,255,255), 3)
            self.visualize(quad_img, f"7.候选{i+1}")

        # 选择最大且符合颜色特征的
        self.final_quad = candidates[0]
        final_img = self.image.copy()
        cv2.drawContours(final_img, [self.final_quad], -1, (255,0,0), 3)
        self.visualize(final_img, "8.最终选择四边形")
        return self.final_quad

    def show_failure(self, message):
        """显示失败信息"""
        fail_img = self.image.copy()
        cv2.putText(fail_img, message, (50,100), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
        self.visualize(fail_img, "ERROR")
        cv2.waitKey(0)

    def analyze(self):
        try:
            # 处理流程
            processed = self.preprocess()
            candidates = self.find_candidates(processed)
            best_quad = self.select_best_quad(candidates)
            
            # 透视变换和颜色识别（此处省略具体实现）
            return {"status": "success"}
            
        except Exception as e:
            if self.show_steps:
                cv2.destroyAllWindows()
            return {"status": "failed", "error": str(e)}

if __name__ == "__main__":
    try:
        detector = VisualCodeDetector("./Sample/Pic2-2.jpg")
        result = detector.analyze()
        print(result)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    except Exception as e:
        print(f"初始化错误: {str(e)}")