import cv2

def  visualize_process(self, title_showd, img_showed):
    """展示图片"""
    if self.show_steps:
        title_showd = "[Step " + str(self.steps_fig) + " ]= " + title_showd
        cv2.imshow(title_showd, img_showed)
        self.steps_fig += 1


def visualization_detect_contours(self):
    """最终识别成果，可视化"""
    if not self.show_steps:
        # 不执行该函数
        return
    
    valid_contours = self.contours
    quadrilaterals = self.quadrilaterals

    # 改进的可视化
    contour_image = cv2.resize(self.image.copy(), (self.target_size_x, self.target_size_y))
    
    # 绘制原始轮廓
    cv2.drawContours(contour_image, valid_contours, -1, (0, 255, 0), 2)
    
    # 绘制四边形（带有效性检查）
    step_vi = 0
    for quad in quadrilaterals:
        step_vi += 1
        # 绘制四边形边界
        cv2.polylines(contour_image, [quad], True, (0, 0, 255), 2)  # 蓝色外接矩形
        
        # 标注坐标点（可选）
        for j, (x, y) in enumerate(quad):
            cv2.circle(contour_image, (x, y), 3, (255, 0, 0), -1)
            cv2.putText(contour_image, f"{j}", (x + 5, y + 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        # 绘制 step 信息
        # 计算四边形的中心点
        x, y, w, h = cv2.boundingRect(quad)
        center_x = x + w // 2
        center_y = y + h // 2

        # 在中心点绘制 step 信息
        cv2.putText(contour_image, f"Step {step_vi}", (center_x - 20, center_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)  # 黑色文本
    
    contour_image2 = cv2.resize(contour_image, (self.target_size_x * 2, self.target_size_y * 2))
    self.visualize_process("Debug View", contour_image2)

def visualization_detect_contours(self):
    """最终识别成果，可视化"""
    if not self.show_steps:
        # 不执行该函数
        return
    
    valid_contours = self.contours
    quadrilaterals = self.quadrilaterals

    # 改进的可视化
    contour_image = cv2.resize(self.image.copy(), (self.target_size_x, self.target_size_y))
    
    # 绘制原始轮廓
    cv2.drawContours(contour_image, valid_contours, -1, (0, 255, 0), 2)
    
    # 绘制四边形（带有效性检查）
    step_vi = 0
    for quad in quadrilaterals:
        step_vi += 1
        # 绘制四边形边界
        cv2.polylines(contour_image, [quad], True, (0, 0, 255), 2)  # 蓝色外接矩形
        
        # 标注坐标点（可选）
        for j, (x, y) in enumerate(quad):
            cv2.circle(contour_image, (x, y), 3, (255, 0, 0), -1)
            cv2.putText(contour_image, f"{j}", (x + 5, y + 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        # 绘制 step 信息
        # 计算四边形的中心点
        x, y, w, h = cv2.boundingRect(quad)
        center_x = x + w // 2
        center_y = y + h // 2

        # 在中心点绘制 step 信息
        cv2.putText(contour_image, f"Step {step_vi}", (center_x - 20, center_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)  # 黑色文本
    
    contour_image2 = cv2.resize(contour_image, (self.target_size_x * 2, self.target_size_y * 2))
    self.visualize_process("Debug View", contour_image2)