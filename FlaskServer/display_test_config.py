"""Android 结果弹窗的手动显示测试配置（伪输出）。

启用后，视频上传仍会正常完成，但后台任务会跳过图像分析，直接向
Android 返回下方配置的测试内容。修改配置后需重启 Flask 服务。

两种模式与真实识别流程的对应关系：

- "record"：第一次识别，记录各阶段时长，Android 显示 T0/T1/T2 与 ΔT1/ΔT2；
- "measure"：再次识别，用已存 ΔT 计算拉伸比和速度，Android 显示
  Name、三个阶段的拉伸百分比、已知 ΔT 与 Speed。

这里的 ``RECORD_TEST_INFO`` / ``MEASURE_TEST_INFO`` 是可直接改数值的样例。
``render_preview()`` 会按 Android 端的排版输出纯文本，便于不接手机也能检查
最终文案。
"""

from copy import deepcopy


# 可选值：
# - None：关闭显示测试，正常分析视频。
# - "record"：模拟第一次记录阶段时间（旧名 "initialization" 仍可用）。
# - "measure"：模拟使用已存时间计算拉伸量和速度（旧名 "preset" 仍可用）。
DISPLAY_TEST_MODE = None

# 弹窗中的视频时长；当前两种显示模板都不展示时长，保留字段以兼容旧接口。
DISPLAY_TEST_DURATION = "12.34"


# DISPLAY_TEST_MODE = "record" 时使用。
# T1/T2 是相邻阶段之间的时长（秒）；ΔT1=T1，ΔT2=T2。
RECORD_TEST_INFO = {
    "mode": "record",
    "stages": {
        "T0": {"time": 0.0, "color": "RGB"},
        "T1": {"time": 1.5, "color": "RG"},
        "T2": {"time": 2.5, "color": "G"},
    },
    "time_differences": {
        "delta_t1": 1.5,
        "delta_t2": 2.5,
        "unit": "s",
    },
}


# DISPLAY_TEST_MODE = "measure" 时使用。
# alpha0/1/2 是 RGB、RG、G 三个阶段的拉伸百分比（100.0 表示 100%）。
# speed.value 由 (alpha2 - alpha0) / 100 × 材料长度 ÷ (ΔT1 + ΔT2) × 60 得到。
MEASURE_TEST_INFO = {
    "mode": "measure",
    "name": "5D code42",
    "stretch_percent": {
        "alpha0": 100.0,
        "alpha1": 127.0,
        "alpha2": 145.0,
    },
    "time_differences": {
        "delta_t1": 1.5,
        "delta_t2": 2.5,
        "unit": "s",
        "known": True,
    },
    "speed": {
        "value": 337.5,
        "unit": "mm/min",
        "material_length_mm": 50.0,
    },
}


_MODE_ALIASES = {
    "record": "record",
    "initialization": "record",
    "measure": "measure",
    "preset": "measure",
}


def _normalize_mode(mode):
    try:
        return _MODE_ALIASES[mode]
    except KeyError as exc:
        raise ValueError(
            "DISPLAY_TEST_MODE must be None, 'record', or 'measure' "
            "('initialization'/'preset' are accepted legacy aliases)"
        ) from exc


def get_display_test_result(mode=None):
    """生成与 /check_status 真实结果相同的测试 result。"""
    selected_mode = DISPLAY_TEST_MODE if mode is None else mode
    if selected_mode is None:
        return None

    normalized = _normalize_mode(selected_mode)
    if normalized == "record":
        info = RECORD_TEST_INFO
    else:
        info = MEASURE_TEST_INFO

    return {
        "duration": str(DISPLAY_TEST_DURATION),
        "info": deepcopy(info),
    }


def _format_seconds(value):
    return "--" if value is None else f"{float(value):.2f}"


def _format_delta(value):
    return "--" if value is None else f"{float(value):.2f}"


def _format_percent(value):
    return "--" if value is None else f"{float(value):.2f}%"


def _format_speed(value, unit):
    return "--" if value is None else f"{float(value):.2f} {unit}"


def render_preview(mode=None):
    """按 Android 端的排版返回弹窗纯文本，供文档和伪输出核对。

    这只是预览，真实排版在 Android 的 ResultDisplayFormatter.kt 中。
    """
    result = get_display_test_result(mode)
    if result is None:
        return None

    info = result["info"]
    lines = ["Image infomation:"]

    if info["mode"] == "record":
        lines.append("Details:")
        for label in ("T0", "T1", "T2"):
            stage = info["stages"][label]
            if label == "T0":
                seconds = "0"
            else:
                seconds = _format_seconds(stage["time"])
            lines.append(f"{label}: {seconds}s {stage['color']}")
        delta = info["time_differences"]
        lines.append(
            f"ΔT1={_format_delta(delta['delta_t1'])}；"
            f"ΔT2={_format_delta(delta['delta_t2'])}；"
        )
        return "\n".join(lines)

    lines.append(f"Name: {info['name']}")
    lines.append("Details:")
    stretch = info["stretch_percent"]
    for label, key in (("RGB α0", "alpha0"), ("RG α1", "alpha1"), ("G α2", "alpha2")):
        lines.append(f"{label}：{_format_percent(stretch.get(key))}")
    delta = info["time_differences"]
    lines.append(
        f"ΔT1={_format_delta(delta['delta_t1'])}；"
        f"ΔT2={_format_delta(delta['delta_t2'])}；（known）"
    )
    lines.append("")
    lines.append("Speed:")
    speed = info["speed"]
    lines.append(_format_speed(speed.get("value"), speed.get("unit", "mm/min")))
    return "\n".join(lines)


def main():
    """在命令行直接查看两种伪输出的最终文案。"""
    for mode in ("record", "measure"):
        print("=" * 40)
        print(f"DISPLAY_TEST_MODE = {mode!r}")
        print("=" * 40)
        print(render_preview(mode))
        print()


if __name__ == "__main__":
    main()
