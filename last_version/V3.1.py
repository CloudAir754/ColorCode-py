import cv2
import numpy as np

# 读取图片
image_path = './Project/Pic.jpg'
image = cv2.imread(image_path)

def ColorDetect(color):
	R,G,B = 0,0,0
	if color[0]> 60:
		R =1
	if color[1]> 60:
		G =1
	if color[2]> 60:
		B =1
	print(R,G,B)
	if R==1 and G==0 and B==0:
		return 'RED'
	if R==0 and G==1 and B==1:
		return 'BLUE'
	if R==1 and G==1 and B==0:
		return 'YELLOW'
	return "WRONG"


# 检查图像是否成功读取
if image is None:
	print("图像读取失败")
else:
	# 调整图片大小
	height , width = image.shape[:2]

	image = cv2.resize(image,(320,320),interpolation=cv2.INTER_AREA)

	# 转换为灰度图像
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	
	# 高斯模糊
	blurred = cv2.GaussianBlur(gray, (5, 5), 0)# 5 5 
	
	# 边缘检测
	edged = cv2.Canny(blurred, 20, 40)# 20 30 
	
	# 形态学变换
	kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)) # 5 5
	closed = cv2.morphologyEx(edged, cv2.MORPH_CLOSE, kernel)
	
	#cv2.imshow("KERNEL",kernel)
	cv2.imshow("CLOS",closed)# 左上色块消失


	# 查找轮廓
	contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	
	# 存储颜色块信息
	colors = []
	
	# 遍历每个轮廓
	for contour in contours:
		# 计算轮廓的边界框
		x, y, w, h = cv2.boundingRect(contour)
		
		# 筛选符合颜色块特征的轮廓
		if 40 < w < 100 and 40 < h < 100:  # 假设颜色块的宽高范围为20到100(示例图像适合40-100)
			# 提取当前轮廓区域
			# grid = image[y:y+h, x:x+w]
			grid = image[int( y+h/4) :int(y+h*3/4) , int(x+w/4):int(x+w*3/4)] # 截取中间区域
			
			# 计算网格的平均颜色
			avg_color = cv2.mean(grid)[:3]
			avg_color = (int(avg_color[2]), int(avg_color[1]), int(avg_color[0]))  # 转换为RGB格式
			# print(avg_color) 
			# 存储颜色信息
			colors.append((x, y, avg_color))
			
			# 绘制边界框（可选）
			cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
	
	# 按位置排序（从上到下，从左到右）
	colors.sort(key=lambda c: (c[1], c[0]))
	
	color_image = [] # 颜色信息记录

	# 输出每个颜色块的颜色信息
	for idx, color in enumerate(colors):
		print(f"颜色块 {idx + 1}: {color[2]}")
		colorback_=ColorDetect(color[2])
		# print(colorback_)
		color_image.append(colorback_)
		# print(color[2][0])
	
	# 显示检测结果
	cv2.imshow('Detected Color Blocks', image)
	cv2.waitKey(0)
	cv2.destroyAllWindows()
	print("Color is :")
	# print(color_image)
	for index, item in enumerate(color_image):
		print(f"{index}: {item}")
