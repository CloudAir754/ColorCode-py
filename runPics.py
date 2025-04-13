import json
import time

from ColorCodeDetector.ColorCodeDetector import ColorCodeDetector

if __name__ == "__main__":
    
    time_start = time.time()

    # v0.3 之后，只需要导入图片
    detector = ColorCodeDetector("./Sample/Sample-01-01.png") # __init__

    result = detector.analyze()
    time_end = time.time()
    print(f"程序识别耗时： {time_end - time_start } ")

    if result.get('Status') == 'Success':
        print("识别结果：")
        for row in result.get('color_matrix', []):
            print(row)
        print("拉伸比例：")
        print(result.get('stretch_ratio'))
    else:
        print(result.get('Error_info'))
