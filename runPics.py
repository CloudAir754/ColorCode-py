import json
import time

from ColorCodeDetector.ColorCodeDetector import ColorCodeDetector

def analyzeSingle(PicPath,pathSwtich=True):
    """
    单张图片分析
        PicPath: 图片路径/图片数组
        pathSwtich: True路径 / False图片数组

    返回：
        "Status":"Success",
        "color_matrix": 3*3数组,
        "stretch_ratio": 拉伸比率,
        "Block_Counts": 块数量,
        "pic_toSave": 图片数组
    """

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
        import cv2
        cv2.imshow("DebugTerminal",result.get('pic_toSave') )
        cv2.waitKey()
        
    else:
        print("出错！")
        print(result.get('Error_info'))
