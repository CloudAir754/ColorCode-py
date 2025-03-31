import json
import time

from ColorCodeDetector.ColorCodeDetector import ColorCodeDetector

if __name__ == "__main__":


    time_start = time.time()
    detector = ColorCodeDetector("./Sample/0331/Pic03_C1-SPEACIAL.png",\
                                 use_provided_quad=False,\
                                    quad_file_path="./ColorCodeDetector/testjson/123.json") # __init__
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
        # 打印错误信息
        print(f"亮度信息：{result.get('Light_Max')}")


