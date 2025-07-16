from ColorCodeDetector.ColorCodeDetector import ColorCodeDetector

PicPath = "./Sample/temp/Ori_"
file = "46"

PicPath = PicPath + file + ".jpg"

picSolve = ColorCodeDetector(PicPath,True)

if __name__ == "__main__":
    picSolve.show_steps = True  # 控制是否显示处理步骤
    picSolve.show_details = False # 展示hsv
    picSolve.HPbrightness_threshold = 120  # 亮度通道阈值，高于此值则加亮度 120 ）
    picSolve.HP_gamma = 0.7  # 指数映射比率 0.7）
    result = picSolve.analyze()    

    print("\n\n\n\n")
    print("="*50)
    print("Recognition result")
    # 格式化拉伸比
    stretch_ratio = round(float(1/result["stretch_ratio"]), 3)
    # stretch_ratio = 2.01 # 用于调整输出
    print(f"Streching radio:     {stretch_ratio}")
       
    # 格式化颜色矩阵为3行
    color_matrix = "\n".join(
        [f"    {str(row):<30}" for row in result['color_matrix']]
    )
    print("Color matrix:")
    print(color_matrix)
    print("="*50)
    print("\n\n\n\n")
    
