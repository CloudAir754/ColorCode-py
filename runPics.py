import json
import time

from ColorCodeDetector.ColorCodeDetector import ColorCodeDetector

def analyzeSingle(PicPath):
    time_start = time.time()

    # v0.3 之后，只需要导入图片
    detector = ColorCodeDetector(PicPath) # __init__

    result = detector.analyze()
    time_end = time.time()
    print(f"程序识别耗时： {time_end - time_start } ")

    return result

if __name__ == "__main__":
    
    result = analyzeSingle("./Sample/Sample-01/Sample-01-03.png")

    if result.get('Status') == 'Success':
        print("识别结果：")
        for row in result.get('color_matrix', []):
            print(row)
        print("拉伸比例：")
        print(result.get('stretch_ratio'))
    else:
        print(result.get('Error_info'))
