from ColorCodeDetector.ColorCodeDetector import ColorCodeDetector

PicPath = "./Sample/temp/Ori_"
file = "44"

PicPath = PicPath + file + ".jpg"

picSolve = ColorCodeDetector(PicPath,True)

if __name__ == "__main__":
    picSolve.show_steps = True  # 控制是否显示处理步骤
    picSolve.show_details = True # 展示hsv
    picSolve.HPbrightness_threshold = 60  # 亮度通道阈值，高于此值则加亮度 120 ）
    picSolve.HP_gamma = 0.3  # 指数映射比率 0.7）
    result = picSolve.analyze()
    print("AAA")