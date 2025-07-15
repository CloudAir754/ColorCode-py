from ColorCodeDetector.ColorCodeDetector import ColorCodeDetector

PicPath = "./Sample/temp/Ori_"
file = "33"

PicPath = PicPath + file + ".jpg"

picSolve = ColorCodeDetector(PicPath,True)

if __name__ == "__main__":
    picSolve.show_steps = True  # 控制是否显示处理步骤
    picSolve.show_details = True # 展示hsv
    result = picSolve.analyze()
    print("AAA")