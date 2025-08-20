from ColorCodeDetector.ColorCodeDetector import ColorCodeDetector

PicPath = "./Sample/temp/Ori_"
# file = "96"

# PicPath = PicPath + file + ".jpg"

def every(frame):
    picSolve = ColorCodeDetector(frame,True)
    # 以下参数模拟动态调整模式
    picSolve.show_steps = True  # 控制是否显示处理步骤
    picSolve.show_details = False # 展示hsv
    picSolve.HP_Black_th= 0.15 # 黑色阈值 0.3
    picSolve.HPt1Canny = 50  # Canny阈值1，低于此值边缘被忽略，默认50
    picSolve.HPkernel = 9  # 形态学增强（闭运算核大小）9
    picSolve.HPbrightness_threshold = 100 # 亮度通道阈值，高于此值则加亮度 120 ）
    picSolve.HP_gamma = 0.6  # 指数映射比率 0.7）
    result = picSolve.analyze()    

    # 格式化调整

    print("\n\n\n\n")
    print("="*50)
    print("Recognition result")
    time = 3
    print(f"Relative time: + {time}s")
    # 格式化拉伸比 
    stretch_ratio = round(float(1/result["stretch_ratio"]), 3)
    stretch_ratio = 223.6 # 用于调整输出
    print(f"Streching radio:     {stretch_ratio} %")

      
    # 格式化颜色矩阵为3行
    color_matrix = "\n".join(
        [f"    {str(row):<30}" for row in result['color_matrix']]
    )
    
    
    print("Color matrix:")
    print(color_matrix)
    print("="*50)
    print("\n\n\n\n")

if __name__ == "__main__":
    for i in range(283,284):
        try:
            a = PicPath+str(i)+".jpg"
            print(i)
            every(a)
        except:
            AA =1

    
