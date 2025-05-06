import json
import time

from ColorCodeDetector.ColorCodeDetector import ColorCodeDetector

def analyzeSingle(PicPath,pathSwtich=True):
    time_start = time.time()

    # v0.3 之后，只需要导入图片
    detector = ColorCodeDetector(PicPath,pathSwtich=pathSwtich) # __init__

    result = detector.analyze()
    time_end = time.time()
    # print(f"程序识别耗时： {time_end - time_start } ")

    return result

if __name__ == "__main__":
    
    result = analyzeSingle("./Sample/Sample-01/Sample-01-03.png")

    if result.get('Status') == 'Success':
        print("识别结果：")
        color_out = result.get('color_matrix', [])
        print(color_out)
        print(f"拉伸比例：{result.get('stretch_ratio'):.2f}")
        print(f"识别到的块数量：{result.get('Block_Counts')}")
        
    else:
        print("出错！")
        print(result.get('Error_info'))
