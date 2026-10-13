"""Android 结果弹窗的手动显示测试配置。

启用后，视频上传仍会正常完成，但后台任务会跳过图像分析，直接向
Android 返回下方配置的测试内容。修改配置后需重启 Flask 服务。
"""

from copy import deepcopy


# 可选值：
# - None：关闭显示测试，正常分析视频。
# - "initialization"：模拟第一次初始化时间预设。
# - "preset"：模拟使用已存时间预设计算拉伸结果。
DISPLAY_TEST_MODE = None

# 弹窗中的视频时长，可以改成任意测试文本。
DISPLAY_TEST_DURATION = "12.34"


# DISPLAY_TEST_MODE = "initialization" 时使用。
# time、description、t1、t2 均可改成想在 Android 弹窗中看到的内容。
INITIALIZATION_TEST_INFO = {
    "stages": {
        "RGB": {"time": "0", "description": "一阶段RGB（自定义测试）"},
        "RG": {"time": "1.50", "description": "二阶段RG（自定义测试）"},
        "G": {"time": "4.00", "description": "三阶段G（自定义测试）"},
    },
    "time_differences": {
        "t1": "1.50",
        "t2": "2.50",
        "unit": "s",
    },
}


# DISPLAY_TEST_MODE = "preset" 时使用。
# name、三个阶段的拉伸文本、t1/t2 以及 display 均可自由修改。
PRESET_TEST_INFO = {
    "name": "5D code01（自定义测试）",
    "stretch": {
        "RGB": "α：100%",
        "RG": "α：127%",
        "G": "α：145%",
    },
    "preset_time_differences": {
        "t1": "1.50",
        "t2": "2.50",
        "unit": "s",
    },
    "stretch_speed": {
        "value": None,
        "unit": "%/s",
        "display": "18.00 %/s（自定义测试）",
    },
}


def get_display_test_result(mode=None):
    """生成与 /check_status 真实结果相同的测试 result。"""
    selected_mode = DISPLAY_TEST_MODE if mode is None else mode
    if selected_mode is None:
        return None
    if selected_mode == "initialization":
        info = INITIALIZATION_TEST_INFO
    elif selected_mode == "preset":
        info = PRESET_TEST_INFO
    else:
        raise ValueError(
            "DISPLAY_TEST_MODE must be None, 'initialization', or 'preset'"
        )

    return {
        "duration": str(DISPLAY_TEST_DURATION),
        "info": deepcopy(info),
    }
