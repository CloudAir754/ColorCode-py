# 待处理的内容
# 【未完成】 3. `morphologyEx`函数的闭合问题
# 6. 拉伸信息检测
# 7. 尝试引入直方图均衡化=test02(放弃)
# 10. 增加指数映射，增加对比度

import cv2
import numpy as np
from sklearn.cluster import KMeans

class ColorCodeDetector:
    def __init__(self, image_path):
        # 初始化图像路径并加载图像
        self.image = cv2.imread(image_path)
        if self.image is None:
            raise ValueError("图像加载失败，请检查文件路径")
        
        
        # 图片
        self.Sized_img = None # 图像大小拉伸后
        self.closed_img = None # 图像(闭运算)

        # 中间元素
        self.contours = []  # 轮廓
        self.quadrilaterals = [] # 有效内接四边形
        self.contours_ordered = [] # 排序后的内接四边形
        self.color_blocks = [] # 颜色快
        self.final_codes = [] # 颜色代码
        self.radio_stretch = 1.0 # 拉伸参数

        # 控制开关
        self.show_steps = True  # 控制是否显示处理步骤
        self.steps_fig =0
        self.show_hsv = True # False 则代表显示rgb
        
        # 参数配置
        self.target_size = 320       # 标准处理尺寸 320               
        self.min_contour_area = 50   # 最小轮廓面积 50
        self.min_screen_coef =  12# 最小有效轮廓占总图像的1/x 8 
        self.max_screen_coef =  3# 最小有效轮廓占总图像的1/x 3

        # 超参数
        self.HPsizeGaussian =5 # 高斯模糊，默认5
        self.HPt1Canny =50 #  Canny阈值1，低于此值边缘被忽略，默认50
        self.HPt2Canny =150 #Canny阈值2，高于此值边缘强边缘，默认150
        self.HPkernel=9 #形态学增强（闭运算核大小）
        self.HP_ts_radio= 0.3 # 拉伸突变容忍参数 
        self.HP_gamma = 0.7 # 指数映射比率
        self.HPbrightness_threshold = 120 # 亮度通道阈值，高于此值则加亮度


    def visualize_process(self,title_showd,img_showed):
        if self.show_steps:
            title_showd="[Step "+str(self.steps_fig)+" ]= "+title_showd
            cv2.imshow(title_showd,img_showed)
            self.steps_fig+=1            
            # 直接展示，再最后退出时再销毁


    def preprocess_image(self):
        """图像预处理流水线，规范大小，找出边缘（初步）"""
        # 尺寸标准化 
        img = cv2.resize(self.image, (self.target_size, self.target_size), 
                        interpolation=cv2.INTER_AREA)
        self.Sized_img = img
        # interpolation=cv2.INTER_AREA 区域插值，适合缩小图像
        
        # 图像增强
        # 转换为 HSV 颜色空间，提取亮度通道（V 通道）
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        v_channel = hsv[:, :, 2]  # 提取 V 通道
        
        # 对亮度较高的像素进行指数映射
        v_normalized = v_channel / 255.0  # 归一化到 [0, 1]
        brightness_threshold = self.HPbrightness_threshold  # 最大的一个分量
        gamma = self.HP_gamma # <0 就是放大
        mask = v_channel > brightness_threshold  # 创建高亮区域的掩码
        v_enhanced = np.where(mask, np.power(v_normalized, gamma) * 255, v_channel)  # 仅增强高亮区域
        v_enhanced = v_enhanced.astype(np.uint8)  # 恢复像素范围
        
        # 将增强后的 V 通道合并回 HSV 图像
        hsv[:, :, 2] = v_enhanced
        img_enhanced = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)  # 转换回 BGR 颜色空间

        gray = cv2.cvtColor(img_enhanced, cv2.COLOR_BGR2GRAY)

        
        # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        blurred = cv2.GaussianBlur(gray, 
                                   (self.HPsizeGaussian,self.HPsizeGaussian), 0) # 高斯模糊
        
        edged = cv2.Canny(blurred, 
                          self.HPt1Canny, self.HPt2Canny) # Canny边缘检测
        
        self.visualize_process("Standard Size Pic",img) # 标准尺寸图片
        self.visualize_process("Gamma Func.",img_enhanced) # 灰度图像
        self.visualize_process("Gray Func.",gray) # 灰度图像
        self.visualize_process("GaussianBlur Func.",blurred) #高斯模糊图像
        self.visualize_process("Canny Edge Func.",edged)# Canny 边缘检测


        # 形态学增强：闭运算以填充边缘检测后的孔洞
        # 原来的核为5*5，改为9*9是为了让部分残缺的方框补全
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT,
                                            (self.HPkernel,self.HPkernel)) # 返回一个卷积核
        closed = cv2.morphologyEx(edged, cv2.MORPH_CLOSE, kernel) # 
        
        #closed 存放的实际是闭环的轮廓的灰度信息
        self.visualize_process("CLOSED Func.",closed)

        self.closed_img = closed
        return closed
    
    def detect_contours(self):
        """这个函数主要作用是剔除无用边缘"""
        contours, _ = cv2.findContours(self.closed_img, 
                                    cv2.RETR_EXTERNAL, 
                                    cv2.CHAIN_APPROX_SIMPLE)
        # contours 存储的是一整圈稀稀落落的点位；
        # 后续处理，会存放在valid_contours列表
        
               
        img_size = self.target_size
        min_dim = img_size // self.min_screen_coef
        max_dim = img_size // self.max_screen_coef
        
        
        valid_contours = [] # 可用边缘
        quadrilaterals = [] # 内接四边形

        for cnt in contours:
            # ===== 新增：轮廓有效性检查 =====
            if len(cnt) < 5:  # 至少需要5个点才能构成有效轮廓
                continue
               
            area = cv2.contourArea(cnt)
            if area < self.min_contour_area:
                continue
                
            # ===== 改进的四边形计算 =====
            try:
                rect = cv2.minAreaRect(cnt) # 用于计算给定点集的最小外接旋转 矩形
                # 这个矩形可以是旋转的，最小化矩形的面积
                '''
                 返回一个 RotatedRect 对象，它包含以下属性：
                    ​center: 矩形的中心坐标 (x, y)。
                    ​size: 矩形的宽度和高度 (width, height)。
                    ​angle: 矩形的旋转角度（以度为单位），范围是 [0, 90)。角度是矩形相对于水平轴的旋转角度。
                '''
                box = cv2.boxPoints(rect)
                '''
                返回一组点，形状为 (4, 2)，表示旋转矩形的四个顶点坐标。
                每个顶点的坐标是 (x, y)
                '''
                box = np.intp(box)  # 更安全的类型转换（替代np.int0）
                
                # 验证四边形坐标
                if (box < 0).any() or (box >= img_size).any():
                    # 这里可能会导致部分情况没有足够的有效边缘
                    continue
            except:
                continue
            
            x, y, w, h = cv2.boundingRect(cnt)
            # 筛选有效轮廓占总图像的1/x
            if not (min_dim < w < max_dim and min_dim < h < max_dim):
                continue

            
            valid_contours.append(cnt) # 有效边缘
            quadrilaterals.append(box) # 有效内接四边形

        # 边缘个数
        print("有效边缘个数：")
        print(len(valid_contours))
        
        print("有效内接四边形个数：")
        print(len(quadrilaterals))
        # cv2.waitKey()
        
        if len(quadrilaterals)!=9:
            print("有效内接四边形不为9！！请检查超参数配置或检查图片")
            cv2.waitKey()
            return []
        print("==============================")



        self.contours = valid_contours 

        self.quadrilaterals = self.sort_quad(quadrilaterals)
        
        

        return quadrilaterals
    

    def sort_quad(self,quadrilaterals):
        """
        将每个四边形的顶点按照左上、右上、右下、左下的顺序排序。

        参数:
            quadrilaterals (list): 包含四边形的列表，每个四边形为 4 个顶点的数组，形状为 (4, 2)。

        返回:
            list: 排序后的四边形列表，每个四边形的顶点顺序为左上、右上、右下、左下。
        """
        sorted_quads = []

        for quad in quadrilaterals:
            # 将四边形顶点转换为 numpy 数组
            quad = np.array(quad)

            # 按照 x 坐标排序
            sorted_by_x = quad[np.argsort(quad[:, 0])]

            # 将排序后的点分为左边两个和右边两个
            left_points = sorted_by_x[:2]
            right_points = sorted_by_x[2:]

            # 在左边两个点中，按照 y 坐标排序（左上、左下）
            left_points = left_points[np.argsort(left_points[:, 1])]
            top_left, bottom_left = left_points[0], left_points[1]

            # 在右边两个点中，按照 y 坐标排序（右上、右下）
            right_points = right_points[np.argsort(right_points[:, 1])]
            top_right, bottom_right = right_points[0], right_points[1]

            # 按照左上、右上、右下、左下的顺序重新排列
            sorted_quad = np.array([top_left, top_right, bottom_right, bottom_left])

            # 添加到结果列表
            sorted_quads.append(sorted_quad)

        # sorted_quadrilaterals =[]
        # 再排序！
        # 这里的排序，四边形中心点在其他中心点上下边之间，他们视为同一行
        #               列同理
        sorted_quadrilaterals=self.sort_quadrilaterals(sorted_quads)







        return sorted_quadrilaterals
    

    def sort_quadrilaterals(self,points_list):
        """
        对 9 个四边形进行排序，按照从左到右、从上到下的顺序排列。
        每个四边形由 4 个点组成，点顺序为 [左上, 右上, 右下, 左下]。
        :param points_l ist: 包含 9 个四边形的列表，每个四边形是 4 个点的列表。
        :return: 排序后的四边形列表。
        """
        # 1. 计算每个四边形的中心点
        quad_centers = []
        for quad in points_list:
            # 中心点的 x 坐标是四个点 x 坐标的平均值
            cx = sum(p[0] for p in quad) / 4
            # 中心点的 y 坐标是四个点 y 坐标的平均值
            cy = sum(p[1] for p in quad) / 4
            quad_centers.append((cx, cy))  # 将中心点添加到列表中
            # print((cx,cy))
        # print("===============")
        # 2. 找出最左上的四边形（A）
        # 最左上的四边形是中心点 x + y 最小的四边形
        a_index = min(range(9), key=lambda i: quad_centers[i][0] + quad_centers[i][1])
        A = points_list[a_index]  # 获取最左上的四边形


        # print(A)
        # print("打印四边形")
        # 3. 计算 A 的顶部和底部边的平均 y 值
        # 顶部边的 y 值是左上和右上点 y 坐标的平均值
        top_avg = (A[0][1] + A[1][1]) / 2
        # 底部边的 y 值是左下和右下点 y 坐标的平均值
        bottom_avg = (A[2][1] + A[3][1]) / 2
        
        # 4. 收集与 A 在同一行的其他两个四边形（B 和 C）
        same_row = []
        for i in range(9):
            if i == a_index:  # 跳过 A 本身
                continue
            cx, cy = quad_centers[i]
            # 如果中心点的 y 值在 A 的顶部和底部之间，说明在同一行
            if top_avg <= cy <= bottom_avg:
                same_row.append((cx, i))  # 记录中心点的 x 坐标和索引
                # print("sssssssssssssssss")

        
        # 5. 按 x 坐标排序，得到 B 和 C 的索引
        same_row.sort(key=lambda x: x[0])  # 按 x 坐标从小到大排序
        B_index = same_row[0][1]  # x 最小的四边形是 B
        C_index = same_row[1][1]  # x 次小的四边形是 C

        
        # 6. 处理剩余的四边形
        remaining = set(range(9)) - {a_index, B_index, C_index}  # 剩下的 6 个四边形
        
        # 7. 计算 B 的左右边界 x 值
        # 左边界是左上和左下点 x 坐标的平均值
        B = points_list[B_index]
        left_x = (B[0][0] + B[3][0]) / 2
        # 右边界是右上和右下点 x 坐标的平均值
        right_x = (B[1][0] + B[2][0]) / 2
        
        # 8. 将剩余四边形分为左、中、右三列
        left_col = []  # 左列
        mid_col = []   # 中列
        right_col = [] # 右列
        for i in remaining:
            cx, cy = quad_centers[i]
            if cx < left_x:  # 如果中心点 x 值小于左边界，属于左列
                left_col.append((cy, i))  # 记录 y 坐标和索引
            elif cx > right_x:  # 如果中心点 x 值大于右边界，属于右列
                right_col.append((cy, i))
            else:  # 否则属于中列
                mid_col.append((cy, i))

        
        
        # 9. 对每列按 y 坐标排序，确定第二行和第三行的四边形
        left_col.sort(key=lambda x: x[0])  # 左列按 y 坐标从小到大排序
        mid_col.sort(key=lambda x: x[0])   # 中列按 y 坐标从小到大排序
        right_col.sort(key=lambda x: x[0]) # 右列按 y 坐标从小到大排序

        if len(left_col)<2 or len(mid_col)<2 or len(right_col)<2:
            print("未足额找到第二行和第三行的方块")
            print(left_col)
            print(mid_col)
            print(right_col)
            print("=======================================")


        # 10. 获取各列的索引，确保索引有效
        indices = [a_index, B_index, C_index]  # 第一行的三个四边形
        
        
        indices.append(left_col[0][1])  # 第二行左列的四边形
        indices.append(mid_col[0][1])  # 第二行中列的四边形
        indices.append(right_col[0][1])  # 第二行右列的四边形

        
        indices.append(left_col[1][1])  # 第三行左列的四边形
        indices.append(mid_col[1][1])  # 第三行中列的四边形
        indices.append(right_col[1][1])  # 第三行右列的四边形
        
        # 11. 按顺序收集所有四边形，跳过无效索引
        ordered = [points_list[i] for i in indices if i is not None]

        self.contours_ordered = ordered
        
        return ordered


    def visualization_detect_contours(self):
        """最终识别成果，可视化"""
        if self.show_steps==False:
            # 不执行该函数
            return
        

        valid_contours = self.contours
        quadrilaterals = self.quadrilaterals

        # ===== 改进的可视化 =====
        contour_image = cv2.resize(self.image.copy(), (self.target_size, self.target_size))
        
        # 绘制原始轮廓
        cv2.drawContours(contour_image, valid_contours, -1, (0, 255, 0), 2)
        
        # 绘制四边形（带有效性检查）
        step_vi =0
        for quad in quadrilaterals:
            step_vi+=1
            # # 最终验证四边形维度？这个没必要吧
            # if quad.shape != (4, 2):
            #     continue
                
            # 绘制四边形边界
            cv2.polylines(contour_image, [quad], True, (0, 0, 255), 2) #蓝色外接矩形
            
            # 标注坐标点（可选）
            for j, (x, y) in enumerate(quad):
                cv2.circle(contour_image, (x, y), 3, (255, 0, 0), -1)
                cv2.putText(contour_image, f"{j}", (x+5, y+5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1)
            

            # 绘制step信息
            # 计算四边形的中心点
            x, y, w, h = cv2.boundingRect(quad)
            center_x = x + w // 2
            center_y = y + h // 2

            # 在中心点绘制 step 信息
            
            cv2.putText(contour_image, f"Step {step_vi}", (center_x - 20, center_y),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)  # 黑色文本
        
        contour_image2= cv2.resize(contour_image,(self.target_size*2,self.target_size*2))
        self.visualize_process("Debug View",contour_image2)

                
        return 
    

       
    def detect_colors(self):
        """颜色检测主逻辑"""
        # 调整图像大小
        detectColor_image = cv2.resize(self.image.copy(), (self.target_size, self.target_size))

        color_means = []  # 存储每个区域的颜色平均值
        color_detects = []  # 存储颜色分类结果

        # 遍历每个内接矩形
        for quad in self.quadrilaterals:
            # 获取内接矩形的顶点坐标
            x, y, w, h = cv2.boundingRect(quad)

            # 计算中心 2/3 面积的区域
            center_factor = 2 / 3
            center_x = int(x + w * (0.5 - center_factor / 2))
            center_y = int(y + h * (0.5 - center_factor / 2))
            center_w = int(w * center_factor)
            center_h = int(h * center_factor)

            # 提取中心区域
            roi = detectColor_image[center_y:center_y + center_h, center_x:center_x + center_w]

            # 根据开关量选择模式
            if self.show_hsv:
                # HSV 模式
                hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
                h_mean = int(np.mean(hsv_roi[:, :, 0]))  # H 通道平均值
                s_mean = int(np.mean(hsv_roi[:, :, 1]))  # S 通道平均值
                v_mean = int(np.mean(hsv_roi[:, :, 2]))  # V 通道平均值
                colors_group = (h_mean, s_mean, v_mean)
                color_means.append(colors_group)
                classify_color_text = self.classify_color(colors_group)
                
                color_detects.append(classify_color_text)

                # 打印 HSV 平均值到终端
                # print(f"Quad at ({x}, {y}, {w}, {h}) - HSV Mean: H={h_mean}, S={s_mean}, V={v_mean}")

                # 在取色区域附近分行显示 HSV 值
                text_position = (center_x, center_y - 30)  # 第一行文本位置
                cv2.putText(detectColor_image, f"H: {h_mean}", text_position,
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)  # 白色文本
                cv2.putText(detectColor_image, f"S: {s_mean}", (center_x, center_y - 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)  # 白色文本
                cv2.putText(detectColor_image, f"V: {v_mean}", (center_x, center_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)  # 白色文本
            else:
                # RGB 模式
                b_mean = int(np.mean(roi[:, :, 0]))  # B 通道平均值
                g_mean = int(np.mean(roi[:, :, 1]))  # G 通道平均值
                r_mean = int(np.mean(roi[:, :, 2]))  # R 通道平均值
                colors_group = (b_mean, g_mean, r_mean)
                color_means.append(colors_group)

                classify_color_text = self.classify_color(colors_group)
                color_detects.append(classify_color_text)

                # 将 RGB 转换为十六进制格式
                hex_color = "#{:02X}{:02X}{:02X}".format(r_mean, g_mean, b_mean)

                # 打印 RGB 平均值到终端
                # print(f"Quad at ({x}, {y}, {w}, {h}) - RGB Mean: B={b_mean}, G={g_mean}, R={r_mean}, Hex: {hex_color}")

                # 在取色区域附近贴十六进制颜色值
                text_position = (center_x, center_y - 10)  # 在取色区域上方显示文本
                cv2.putText(detectColor_image, hex_color, text_position,
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)  # 白色文本

            # 在取色区域附近显示颜色分类结果
            classify_text_position = (center_x, center_y + center_h + 20)  # 在取色区域下方显示分类结果
            cv2.putText(detectColor_image, classify_color_text, classify_text_position,
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)  # 绿色文本

            # 可视化取色区域
            cv2.rectangle(detectColor_image, (center_x, center_y),
                        (center_x + center_w, center_y + center_h), (0, 255, 0), 2)  # 绿色矩形框

        # 调整图像大小以便显示
        detectColor_image2 = cv2.resize(detectColor_image.copy(), (self.target_size * 2, self.target_size * 2))

        # 显示结果图像
        self.visualize_process("Color Detect",detectColor_image2)


        self.color_blocks = color_means  # 原始颜色块信息
        self.final_codes = color_detects  # 颜色分类结果

        return color_means
                
    def classify_color(self,color):
        """
        根据颜色代码判断颜色类别（红色、蓝色、绿色、黑色）。
        
        参数:
            color (tuple): 颜色代码，为 (H, S, V) 或 (B, G, R) 元组。
            use_hsv (bool): 是否使用 HSV 颜色空间，默认为 True。如果为 False，则使用 RGB 颜色空间。
        
        返回:
            str: 颜色类别，例如 "红色"、"蓝色"、"绿色"、"黑色" 或 "未知颜色"。
        """

        if self.show_hsv:
            # HSV 模式
            # HSV色盘 https://lab.pyzy.net/palette.html
            h, s, v = color
            h=h*2
            # 不考虑黑色，取左不取右
            From0 = 0
            From0_endRed = 60
            FromRed_endGreen = 165
            FromGreen_endBlue = 300
            FromBlue_end360 = 360
            Black_below = 60

            if v<Black_below:
                return "Black"

            if From0 <= h < From0_endRed:
                return "Red"
            elif  h < FromRed_endGreen:
                return "Green"
            elif  h < FromGreen_endBlue:
                return "Blue"
            elif  h < FromBlue_end360:
                return "Red"

            return f"H:{h},S:{s},V:{v}"
        else:
            # RGB 模式
            b, g, r = color
            return f"B:{b},G:{g},R:{r}"
        
    def detect_stretch_ratio(self):
        """检测拉伸比率，并检查波动是否过大。"""
        if not self.contours_ordered:
            raise ValueError("没有检测到有效的四边形，无法计算拉伸比率。")
        
        ratios = []
        
        # 计算每个四边形的长宽比
        for quad in self.contours_ordered:
            # 计算四边形的宽度和高度
            width = np.linalg.norm(quad[0] - quad[1])  # 左上到右上的距离
            height = np.linalg.norm(quad[1] - quad[2])  # 右上到右下的距离
            
            # 计算长宽比 x/y
            ratio = width / height
            ratios.append(ratio)
        
        # 计算平均长宽比
        mean_ratio = np.mean(ratios)
        
        # 检查每个比率与均值的差异
        for i, ratio in enumerate(ratios):
            deviation = abs(ratio - mean_ratio) / mean_ratio
            if deviation > self.HP_ts_radio:
                print((f"第 {i+1} 个四边形的拉伸比率 {ratio:.2f} 与均值 {mean_ratio:.2f} 的差异超过阈值 {self.HP_ts_radio*100:.0f}%。"))
                # raise ValueError(f"第 {i+1} 个四边形的拉伸比率 {ratio:.2f} 与均值 {mean_ratio:.2f} 的差异超过阈值 {self.HP_ts_radio*100:.0f}%。")
        
        self.radio_stretch = mean_ratio
        return mean_ratio



    


    
    def analyze(self):
        """完整处理流程"""

        try:
            # 执行图像预处理、轮廓检测、颜色检测和可视化
            
            self.preprocess_image() # 预处理
            self.detect_contours() # 轮廓检测
            radio_stretch = self.detect_stretch_ratio() # 检测拉伸比率
            print(f"平均拉伸比率: {radio_stretch:.2f}")
            
            self.detect_colors() # 颜色检测
            

            color_matrix = self.final_codes

            self.visualization_detect_contours()# 可视化找色块
            
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